from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
#from tensorflow.python.ops import array_ops

from utils import *

def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims

'''
_cuda_op_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel.so'))
zbuffer_tri = _cuda_op_module.zbuffer_tri

def ZBuffer_Rendering_CUDA_op(s2d, tri):
    tri_map, zbuffer = zbuffer_tri(s2d, tri)
    #mask = tf.cast(tf.not_equal(zbuffer, tf.zeros_like(zbuffer)), tf.float32)
    return tri_map, zbuffer

ops.NotDifferentiable("ZbufferTri")
'''

'''
_cuda_op_module_v2 = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel_v2.so'))
zbuffer_tri_v2 = _cuda_op_module_v2.zbuffer_tri_v2

def ZBuffer_Rendering_CUDA_op_v2(s2d, tri, vis):
    tri_map, zbuffer = zbuffer_tri_v2(s2d, tri, vis)
    #mask = tf.cast(tf.not_equal(zbuffer, tf.zeros_like(zbuffer)), tf.float32)
    return tri_map, zbuffer

ops.NotDifferentiable("ZbufferTriV2")
'''

_cuda_op_module_v2_sz224 = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel_v2_sz224.so'))
zbuffer_tri_v2_sz224 = _cuda_op_module_v2_sz224.zbuffer_tri_v2_sz224

def ZBuffer_Rendering_CUDA_op_v2_sz224(s2d, tri, vis):
    tri_map, zbuffer = zbuffer_tri_v2_sz224(s2d, tri, vis)
    return tri_map, zbuffer
ops.NotDifferentiable("ZbufferTriV2Sz224")

'''
_cuda_op_module_v3 = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'TF_newop/cuda_op_kernel_v3.so'))
zbuffer_tri_v3 = _cuda_op_module_v3.zbuffer_tri_v3

def ZBuffer_Rendering_CUDA_op_v3(s2d, tri, vis):
    tri_map, zbuffer, coord1, coord2, coord3 = zbuffer_tri_v3(s2d, tri, vis)
    #mask = tf.cast(tf.not_equal(zbuffer, tf.zeros_like(zbuffer)), tf.float32)
    return tri_map, zbuffer, coord1, coord2, coord3

ops.NotDifferentiable("ZbufferTriV3")
'''

#@ops.RegisterGradient("ZbufferTri")
#def _zero_out_grad(op, grad):
 
#  s2d = op.inputs[0]
#  s2d_shape = array_ops.shape(s2d)
#  s2d_grad = array_ops.zeros_like(s2d_shape)

#  tri = op.inputs[0]
#  tri_shape = array_ops.shape(tri)
#  tri_grad = array_ops.zeros_like(tri_shape)

#  return [s2d_grad, tri_grad]

def unwarp_texture(image, m, mshape, output_size=124, is_reduce = False):
    #TO Do: correct the mask
    print("TODO: correct the mask in unwarp_texture(image, m, mshape, output_size=124, is_reduce = False)")

    def flatten(x):
        return tf.reshape(x, [-1])


    n_size = get_shape(image)
    n_size = n_size[0]
    s = output_size   

     # Tri, tri2vt
    if is_reduce:
        tri = load_3DMM_tri_reduce()
        vertex_tri = load_3DMM_vertex_tri_reduce()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()

        tri_2d = load_FaceAlignment_tri_2d_reduce()
    else:
        tri = load_3DMM_tri()
        vertex_tri = load_3DMM_vertex_tri()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()

        tri_2d = load_FaceAlignment_tri_2d()
        
    tri2vt1_const = tf.constant(tri[0,:], tf.int32)
    tri2vt2_const = tf.constant(tri[1,:], tf.int32)
    tri2vt3_const = tf.constant(tri[2,:], tf.int32)

    tri_const = tf.constant(tri, tf.int32)
    tri_2d_const = tf.constant(tri_2d, tf.int32)
    tri_2d_const_flat = flatten(tf.constant(tri_2d, tf.int32))

    vertex_tri_const = tf.constant(vertex_tri, tf.int32)


    #Vt2pix
    vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
    vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)

    #indicies = np.zeros([s*s,2])
    #for i in range(s):
    #    for j in range(s):
    #        indicies[i*s+j ] = [i,j]

    #indicies_const = tf.constant(indicies, tf.float32)
    #[indicies_const_u, indicies_const_v] = tf.split(1, 2, indicies_const)

    ###########m = m * tf.constant(self.std_m) + tf.constant(self.mean_m)
    ###########mshape = mshape * tf.constant(self.std_shape) + tf.constant(self.mean_shape)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
    
    pixel_u = []
    pixel_v = []

    masks = []
    for i in range(n_size):

        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)

        # Compute 2d vertex
        #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))



        normal, normalf = compute_normal(vertex3d_rs,tri_const, vertex_tri_const)
        normalf = tf.transpose(normalf)
        normalf4d = tf.concat(axis=0, values=[normalf, tf.ones([1, normalf.get_shape()[-1]], tf.float32)])
        rotated_normalf = tf.matmul(m_i, normalf4d, False, False)
        _, _, rotated_normalf_z = tf.split(axis=0, num_or_size_splits=3, value=rotated_normalf)
        visible_tri = flatten(tf.greater(rotated_normalf_z, 0))

        #print("get_shape(visible_tri)")
        #print(get_shape(visible_tri))
        mask_i = tf.gather( tf.cast(visible_tri, dtype=tf.float32),  tri_2d_const_flat )
        #print("get_shape(mask_i)")
        #print(get_shape(mask_i))
        mask_i = tf.reshape( mask_i, tri_2d.shape)
        #print("get_shape(mask_i)")
        #print(get_shape(mask_i))


        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, vertex3d_rs.get_shape()[-1]], tf.float32)])
        
        vertex2d = tf.matmul(m_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = tf.squeeze(vertex2d_u - 2)
        vertex2d_v = tf.squeeze(s - vertex2d_v - 1)

        #vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
        #vertex2d = tf.transpose(vertex2d)

        #vertex2d_u = tf.transpose(vertex2d_u)
        #vertex2d_V = tf.transpose(vertex2d_v)



        vt1 = tf.gather( tri2vt1_const,  tri_2d_const ) 
        vt2 = tf.gather( tri2vt2_const,  tri_2d_const ) 
        vt3 = tf.gather( tri2vt3_const,  tri_2d_const )

        


        pixel1_u = tf.gather( vertex2d_u,  vt1 ) #tf.gather( vt2pixel_u_const,  vt1 ) 
        pixel2_u = tf.gather( vertex2d_u,  vt2 ) 
        pixel3_u = tf.gather( vertex2d_u,  vt3 )

        pixel1_v = tf.gather( vertex2d_v,  vt1 ) 
        pixel2_v = tf.gather( vertex2d_v,  vt2 ) 
        pixel3_v = tf.gather( vertex2d_v,  vt3 )

        pixel_u_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_u, pixel2_u, pixel3_u]))
        pixel_v_i = tf.scalar_mul(scalar = 1.0/3.0, x = tf.add_n([pixel1_v, pixel2_v, pixel3_v]))

        pixel_u.append(pixel_u_i)
        pixel_v.append(pixel_v_i)


        masks.append(mask_i)


        
    texture = bilinear_sampler(image, pixel_u, pixel_v)
    masks = tf.stack(masks)

    return texture, masks

def warp_texture(texture, m, mshape, output_size=96, is_reduce = False):
    pixel_u, pixel_v, masks = warping_flow(m, mshape, output_size, is_reduce)
    images = bilinear_sampler(texture, pixel_v, pixel_u)

    return images, masks



def warping_flow(m, mshape, output_size=96, is_reduce = False):
    def flatten(x):
        return tf.reshape(x, [-1])


    n_size = get_shape(m)
    n_size = n_size[0]

    s = output_size   

    # Tri, tri2vt
    if is_reduce:
        tri = load_3DMM_tri_reduce()
        vertex_tri = load_3DMM_vertex_tri_reduce()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
    else:
        tri = load_3DMM_tri()
        vertex_tri = load_3DMM_vertex_tri()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()
        
    tri2vt1_const = tf.constant(tri[0,:], tf.int32)
    tri2vt2_const = tf.constant(tri[1,:], tf.int32)
    tri2vt3_const = tf.constant(tri[2,:], tf.int32)

    tri_const = tf.constant(tri, tf.int32)
    vertex_tri_const = tf.constant(vertex_tri, tf.int32)


    #Vt2pix
    vt2pixel_u_const = tf.constant(vt2pixel_u, tf.float32)
    vt2pixel_v_const = tf.constant(vt2pixel_v, tf.float32)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
    
    pixel_u = []
    pixel_v = []

    masks = []

    u, v = tf.meshgrid( tf.linspace(0.0, output_size-1.0, output_size), tf.linspace(0.0, output_size-1.0, output_size))
    u = flatten(u)
    v = flatten(v)

    for i in range(n_size):

        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)

        # Compute 2d vertex
        #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))

        normal, normalf = compute_normal(vertex3d_rs,tri_const, vertex_tri_const)
        normalf = tf.transpose(normalf)
        normalf4d = tf.concat(axis=0, values=[normalf, tf.ones([1, normalf.get_shape()[-1]], tf.float32)])
        rotated_normalf = tf.matmul(m_i, normalf4d, False, False)
        _, _, rotated_normalf_z = tf.split(axis=0, num_or_size_splits=3, value=rotated_normalf)
        visible_tri = tf.greater(rotated_normalf_z, 0)


        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, vertex3d_rs.get_shape()[-1]], tf.float32)])
        
        vertex2d = tf.matmul(m_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = vertex2d_u - 1
        vertex2d_v = s - vertex2d_v

        vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
        vertex2d = tf.transpose(vertex2d)

        if output_size == 96:
            tri_map_2d, mask_i = ZBuffer_Rendering_CUDA_op_v2(vertex2d, tri_const, visible_tri)
        else:
            tri_map_2d, mask_i = ZBuffer_Rendering_CUDA_op_v2_sz224(vertex2d, tri_const, visible_tri)

        #tri_map_2d, mask_i, coord1, coord2, coord3 = ZBuffer_Rendering_CUDA_op_v3(vertex2d, tri_const, visible_tri)
        

        tri_map_2d_flat = tf.cast(tf.reshape(tri_map_2d, [-1]), 'int32')
        

        # Calculate barycentric coefficient
        
        vt1 = tf.gather( tri2vt1_const,  tri_map_2d_flat ) 
        vt2 = tf.gather( tri2vt2_const,  tri_map_2d_flat ) 
        vt3 = tf.gather( tri2vt3_const,  tri_map_2d_flat )

        
        pixel1_uu = flatten(tf.gather( vertex2d_u,  vt1 ))
        pixel2_uu = flatten(tf.gather( vertex2d_u,  vt2 ))
        pixel3_uu = flatten(tf.gather( vertex2d_u,  vt3 ))

        pixel1_vv = flatten(tf.gather( vertex2d_v,  vt1 ))
        pixel2_vv = flatten(tf.gather( vertex2d_v,  vt2 ))
        pixel3_vv = flatten(tf.gather( vertex2d_v,  vt3 ))
        c1, c2, c3 = barycentric(pixel1_uu, pixel2_uu, pixel3_uu, pixel1_vv, pixel2_vv, pixel3_vv, u, v)
        
        #c1 = tf.constant(1/3, dtype=tf.float32)
        #c2 = tf.constant(1/3, dtype=tf.float32)
        #c3 = tf.constant(1/3, dtype=tf.float32)

        
        ##
        pixel1_u = tf.gather( vt2pixel_u_const,  vt1 ) 
        pixel2_u = tf.gather( vt2pixel_u_const,  vt2 ) 
        pixel3_u = tf.gather( vt2pixel_u_const,  vt3 )

        pixel1_v = tf.gather( vt2pixel_v_const,  vt1 ) 
        pixel2_v = tf.gather( vt2pixel_v_const,  vt2 ) 
        pixel3_v = tf.gather( vt2pixel_v_const,  vt3 )


        pixel_u_i = tf.reshape(pixel1_u * c1 + pixel2_u * c2 + pixel3_u* c3, [output_size, output_size])
        pixel_v_i = tf.reshape(pixel1_v * c1 + pixel2_v * c2 + pixel3_v* c3, [output_size, output_size])


        pixel_u.append(pixel_u_i)
        pixel_v.append(pixel_v_i)

        masks.append(mask_i)
        
    #images = bilinear_sampler(texture, pixel_v, pixel_u)
    masks = tf.stack(masks)

    return pixel_u, pixel_v, masks

def barycentric(pixel1_u, pixel2_u, pixel3_u, pixel1_v, pixel2_v, pixel3_v, u, v):

    v0_u = pixel2_u - pixel1_u
    v0_v = pixel2_v - pixel1_v

    v1_u = pixel3_u - pixel1_u
    v1_v = pixel3_v - pixel1_v

    v2_u = u - pixel1_u
    v2_v = v - pixel1_v

    invDenom = 1.0/(v0_u * v1_v - v1_u * v0_v + 1e-6)
    c2 = (v2_u * v1_v - v1_u * v2_v) * invDenom
    c3 = (v0_u * v2_v - v2_u * v0_v) * invDenom
    c1 = 1.0 - c2 - c3

    return c1, c2, c3


def shading(L, normal):

    
    shape = normal.get_shape().as_list()
    
    normal_x, normal_y, normal_z = tf.split(tf.expand_dims(normal, -1), axis=2, num_or_size_splits=3)
    pi = math.pi

    sh=[0]*9
    sh[0] = 1/math.sqrt(4*pi) * tf.ones_like(normal_x)
    sh[1] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_z
    sh[2] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_y
    sh[3] = ((2*pi)/3)*(math.sqrt(3/(4*pi)))* normal_x
    sh[4] = (pi/4)*(1/2)*(math.sqrt(5/(4*pi)))*(2*tf.square(normal_z)-tf.square(normal_x)-tf.square(normal_y))
    sh[5] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_y*normal_z)
    sh[6] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_z)
    sh[7] = (pi/4)*(3)  *(math.sqrt(5/(12*pi)))*(normal_x*normal_y)
    sh[8] = (pi/4)*(3/2)*(math.sqrt(5/(12*pi)))*( tf.square(normal_x)-tf.square(normal_y))

    sh = tf.concat(sh, axis=3)
    print('sh.get_shape()')
    print(sh.get_shape())

    if L.get_shape()[-1] == 34:
        Ls, Lp = tf.split(L, [27, 7], axis=1)

        # Specular shading
        incoming_light, outgoing_light, shininess = tf.split(Lp, [3, 3, 1], axis=1)

        incoming_light = tf.nn.l2_normalize(incoming_light, 1)
        incoming_light = tf.expand_dims(incoming_light, 1)
        incoming_light = tf.tile(incoming_light, multiples=[1, shape[1], 1] )
        incoming_light = tf.expand_dims(incoming_light, 2)
        print('incoming_light.get_shape()')
        print(incoming_light.get_shape())

        outgoing_light = tf.nn.l2_normalize(outgoing_light, 1)
        outgoing_light = tf.expand_dims(outgoing_light, 1)
        outgoing_light = tf.tile(outgoing_light, multiples=[1, shape[1], 1] )
        outgoing_light = tf.expand_dims(outgoing_light, -1)
        print('outgoing_light.get_shape()')
        print(outgoing_light.get_shape())

        print('normal.get_shape()')
        print(normal.get_shape())

        dot_product = tf.matmul(incoming_light, tf.expand_dims(normal, -1))
        perfect_reflection_direction = incoming_light-2.0*dot_product*tf.expand_dims(normal, 2)
        perfect_reflection_direction = tf.nn.l2_normalize(perfect_reflection_direction, -1)

        print('perfect_reflection_direction.get_shape()')
        print(perfect_reflection_direction.get_shape())

        cos_alpha = tf.matmul(perfect_reflection_direction, outgoing_light)
        print('cos_alpha.get_shape()')
        print(cos_alpha.get_shape())

        #cos_alpha = tf.clip_by_value(cos_alpha, 0, 1)   
        cos_alpha = tf.maximum(cos_alpha, tf.zeros_like(cos_alpha))
        cos_alpha = tf.squeeze(cos_alpha, -1)

        shininess = tf.maximum(shininess, tf.zeros_like(shininess))


        shading_specular = tf.pow(cos_alpha, tf.tile(tf.expand_dims(shininess,1), multiples=[1, shape[1], 1] ))

        #shading_specular = tf.clip_by_value(shading_specular, 0, 5)   



    else:
        Ls = L
        shading_specular = 0


    L1, L2, L3 = tf.split(Ls, num_or_size_splits = 3, axis=1)
    L1 = tf.expand_dims(L1, 1)
    L1 = tf.tile(L1, multiples=[1, shape[1], 1] )
    L1 = tf.expand_dims(L1, -1)

    L2 = tf.expand_dims(L2, 1)
    L2 = tf.tile(L2, multiples=[1, shape[1], 1] )
    L2 = tf.expand_dims(L2, -1)

    L3 = tf.expand_dims(L3, 1)
    L3 = tf.tile(L3, multiples=[1, shape[1], 1] )
    L3 = tf.expand_dims(L3, -1)

    print('L1.get_shape()')
    print(L1.get_shape())

    B1 = tf.matmul(sh, L1)
    B2 = tf.matmul(sh, L2)
    B3 = tf.matmul(sh, L3)

    B = tf.squeeze(tf.concat([B1, B2, B3], axis = 2))



    return 0.75*B + 0.25*shading_specular

def generate_shade(il, m, mshape, texture_size = [192, 224], is_reduce = False, is_with_normal=False):
    '''
    print("get_shape(il) ")
    print(get_shape(il) )
    print("get_shape(m) ")
    print(get_shape(m) )
    print("get_shape(mshape) ")
    print(get_shape(mshape) )
    '''

    n_size = get_shape(il)       
    n_size = n_size[0]

    # Tri, tri2vt
    if is_reduce:
        tri = load_3DMM_tri_reduce()
        vertex_tri = load_3DMM_vertex_tri_reduce()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
        tri_2d = load_FaceAlignment_tri_2d_reduce()
        tri_2d_barycoord = load_FaceAlignment_tri_2d_barycoord_reduce()
    else:
        tri = load_3DMM_tri()
        vertex_tri = load_3DMM_vertex_tri()
        vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()
        tri_2d = load_FaceAlignment_tri_2d()
        
    
    tri_const = tf.constant(tri, tf.int32)
    vertex_tri_const = tf.constant(vertex_tri, tf.int32)

    
    tri_2d_const = tf.constant(tri_2d, tf.int32)
    tri_2d_const_flat = tf.reshape(tri_2d_const, shape=[-1,1])

    tri2vt1_const = tf.constant(tri[0,:], tf.int32)
    tri2vt2_const = tf.constant(tri[1,:], tf.int32)
    tri2vt3_const = tf.constant(tri[2,:], tf.int32)

    vt1 = tf.gather( tri2vt1_const,  tri_2d_const_flat ) 
    vt2 = tf.gather( tri2vt2_const,  tri_2d_const_flat ) 
    vt3 = tf.gather( tri2vt3_const,  tri_2d_const_flat )

    vt1_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:,:,0], tf.float32), shape=[-1,1])
    vt2_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:,:,1], tf.float32), shape=[-1,1])
    vt3_coeff = tf.reshape(tf.constant(tri_2d_barycoord[:,:,2], tf.float32), shape=[-1,1])



    #mshape = mshape * tf.constant(self.std_shape) + tf.constant(self.mean_shape)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)

    #def get_normal_flat(shape_single):
    #    vertex3d_rs = tf.transpose(tf.reshape( shape_single, shape = [-1, 3] ))
    #    normal, normalf = compute_normal(vertex3d_rs, tri_const, vertex_tri_const)
    #    normalf_flat = tf.gather_nd(normalf, tri_2d_const_flat)
    #    normalf_flats.append(normalf_flat)

  
    #normalf_flats = tf.map_fn( lambda ss: get_normal_flat(ss), shape_single  )
    
    normalf_flats = []
    for i in range(n_size):
        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.cross(m_i_row1, m_i_row2)
        m_i = tf.concat([ tf.expand_dims(m_i_row1, 0), tf.expand_dims(m_i_row2, 0), tf.expand_dims(m_i_row3, 0)], axis = 0)




        '''
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)
        print('m_i.shape()')
        print(m_i.get_shape())
        '''

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))

        normal, normalf = compute_normal(vertex3d_rs, tri_const, vertex_tri_const)


        ###
        '''
        normalf = tf.transpose(normalf)
        rotated_normalf = tf.matmul(m_i, normalf, False, False)
        rotated_normalf = tf.transpose(rotated_normalf)

        normalf_flat = tf.gather_nd(rotated_normalf, tri_2d_const_flat) 
        normalf_flats.append(normalf_flat)
        '''




        ###
        normal = tf.transpose(normal)
        rotated_normal = tf.matmul(m_i, normal, False, False)
        rotated_normal = tf.transpose(rotated_normal)
        normal_flat_vt1 = tf.gather_nd(rotated_normal, vt1)
        normal_flat_vt2 = tf.gather_nd(rotated_normal, vt2)
        normal_flat_vt3 = tf.gather_nd(rotated_normal, vt3)
        
        normalf_flat = normal_flat_vt1*vt1_coeff + normal_flat_vt2*vt2_coeff + normal_flat_vt3*vt3_coeff
        normalf_flats.append(normalf_flat)




    normalf_flats = tf.stack(normalf_flats)
    
    #print("normalf_flats.get_shape()")
    #print(normalf_flats.get_shape())

    #print("il.get_shape()")
    #print(il.get_shape())

    shade = shading(il, normalf_flats)

    #print("shade.get_shape()")
    #print(shade.get_shape())

    if is_with_normal:
        return tf.reshape(shade, shape = [-1, texture_size[0], texture_size[1], 3]), tf.reshape(normalf_flats, shape = [-1, texture_size[0], texture_size[1], 3]), 



    return tf.reshape(shade, shape = [-1, texture_size[0], texture_size[1], 3])



def compute_normal(vertex, tri, vertex_tri):
    # Unit normals to the faces
    # vertex : 3xvertex_num
    # tri : 3xtri_num

    '''
    print("get_shape(vertex)")
    print(get_shape(vertex))
    print("get_shape(tri)")
    print(get_shape(tri))
    print("get_shape(vertex_tri)")
    print(get_shape(vertex_tri))
    '''



    vertex = tf.transpose(vertex)

    vt1_indices, vt2_indices, vt3_indices = tf.split(tf.transpose(tri), num_or_size_splits = 3, axis = 1)

    #print("vertex.get_shape()")
    #print(vertex.get_shape())
    

    vt1 = tf.gather_nd(vertex, vt1_indices)
    vt2 = tf.gather_nd(vertex, vt2_indices)
    vt3 = tf.gather_nd(vertex, vt3_indices)

    #print(vt1.get_shape())
    #print("vt1.get_shape()")

    normalf = tf.cross(vt2 - vt1, vt3 - vt1)
    normalf = tf.nn.l2_normalize(normalf, dim = 1)
    #print(normalf.get_shape())
    #print("normalf.get_shape()")
    
    #print(tri.shape)
    #print(105840)
    #TRI_NUM = 105840
    mask = tf.tile( tf.expand_dims(  tf.not_equal(vertex_tri, tri.shape[1] - 1), 2), multiples = [1, 1, 3])
    mask = tf.cast( mask, vertex.dtype  )
    vertex_tri = tf.reshape(vertex_tri, shape = [-1, 1])
    normal = tf.reshape(tf.gather_nd(normalf, vertex_tri), shape = [8, -1, 3])

    
    #print("normal.get_shape()")
    #print(normal.get_shape())

    normal = tf.reduce_sum( tf.multiply( normal, mask ),  axis = 0)
    #count =  tf.reduce_sum( mask,  axis = 0)
    #normal = tf.divide(normal, count)
    normal = tf.nn.l2_normalize(normal, dim = 1)



    #print("normal.get_shape()")
    #print(normal.get_shape())


    # enforce that the normal are outward
    v = vertex - tf.reduce_mean(vertex,0)
    s = tf.reduce_sum( tf.multiply(v, normal), 0 )

    count_s_greater_0 = tf.count_nonzero( tf.greater(s, 0) )
    count_s_less_0 = tf.count_nonzero( tf.less(s, 0) )

    sign = 2 * tf.cast(tf.greater(count_s_greater_0, count_s_less_0), tf.float32) - 1
    normal = tf.multiply(normal, sign)
    normalf = tf.multiply(normalf, sign)

    return normal, normalf

def compute_tri_normal(vertex,tri, vertex_tri):
    # Unit normals to the faces
    # vertex : 3xvertex_num
    # tri : 3xtri_num

    vertex = tf.transpose(vertex)

    vt1_indices, vt2_indices, vt3_indices = tf.split(tf.transpose(tri), num_or_size_splits = 3, axis = 1)

    vt1 = tf.gather_nd(vertex, vt1_indices)
    vt2 = tf.gather_nd(vertex, vt2_indices)
    vt3 = tf.gather_nd(vertex, vt3_indices)

    normalf = tf.cross(vt2 - vt1, vt3 - vt1)
    normalf = tf.nn.l2_normalize(normalf, dim = 1)

    return normalf

compute_normal2 = compute_tri_normal


def compute_landmarks(m, shape, output_size=224, is_reduce = False, is_3d = False):
    # m: rotation matrix [batch_size x (4x2)]
    # shape: 3d vertices location [batch_size x (vertex_num x 3)]

    n_size = get_shape(m)    
    n_size = n_size[0]

    s = output_size   


    # Tri, tri2vt
    if is_reduce:
        kpts = load_3DMM_kpts_reduce()
    else:
        kpts = load_3DMM_kpts()

    kpts_num = kpts.shape[0]

    indices = np.zeros([n_size, kpts_num,2], np.int32)
    for i in range(n_size):
        indices[i,:,0] = i
        indices[i,:,1:2] = kpts

    indices = tf.constant(indices, tf.int32)

    kpts_const = tf.constant(kpts, tf.int32)

    vertex3d = tf.reshape( shape, shape = [n_size, -1, 3] )                                                   # batch_size x vertex_num x 3
    vertex3d = tf.gather_nd(vertex3d, indices)        # Keypoints selection                                   # batch_size x kpts_num x 3
    vertex4d = tf.concat(axis = 2, values = [vertex3d, tf.ones(get_shape(vertex3d)[0:2] +[1], tf.float32)])   # batch_size x kpts_num x 4

    m = tf.reshape( m, shape = [n_size, 4, 2] )
    if is_3d:
        m_row1 = tf.nn.l2_normalize(m[:,0:3,0], axis = 1)
        m_row2 = tf.nn.l2_normalize(m[:,0:3,1], axis = 1)

        m_row1_norm = tf.norm(m[:,0:3,0], ord='euclidean', axis = 1)
        m_row2_norm = tf.norm(m[:,0:3,1], ord='euclidean', axis = 1)
        m_row3_norm = (m_row1_norm + m_row2_norm)/2



        m_row3 = tf.pad(  tf.expand_dims(m_row3_norm, 1) * tf.cross(m_row1, m_row2), [[0,0],[0,1]], mode='CONSTANT', constant_values=0)
        m_row3 = tf.expand_dims(m_row3, axis=2)

        m = tf.concat([m, m_row3], axis = 2)

    vertex2d = tf.matmul(m, vertex4d, True, True)                                                             # batch_size x 2 [or 3] x kpts_num
    vertex2d = tf.transpose(vertex2d, perm=[0,2,1])                                                           # batch_size x kpts_num x 2 [or 3]

    if is_3d:
        [vertex2d_u, vertex2d_v, vertex2d_z]  = tf.split(axis=2, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = vertex2d_u - 1 
        vertex2d_v = s - vertex2d_v

        return vertex2d_u, vertex2d_v, vertex2d_z

    else:
        [vertex2d_u, vertex2d_v]  = tf.split(axis=2, num_or_size_splits=2, value=vertex2d)
        vertex2d_u = vertex2d_u - 1 
        vertex2d_v = s - vertex2d_v

        return vertex2d_u, vertex2d_v

'''
def compute_landmarks(m, mshape, output_size=224, is_reduce = False):

    n_size = get_shape(m)    
    n_size = n_size[0]

    s = output_size   


    # Tri, tri2vt
    if is_reduce:
        kpts = load_3DMM_kpts_reduce()
    else:
        kpts = load_3DMM_kpts()

    kpts_const = tf.constant(kpts, tf.int32)

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)

    landmarks_u = []
    landmarks_v = []

    for i in range(n_size):
        # Compute 2d vertex
        #vertex3d = tf.transpose(tf.reshape( mu_const + tf.matmul(w_shape_const, p_shape_single[i], False, True) + tf.matmul(w_exp_const, p_exp_single[i], False, True), shape = [-1, 3] ))

        vertex3d_rs = tf.reshape( shape_single[i], shape = [-1, 3] )
        vertex3d_rs = tf.transpose(tf.gather_nd(vertex3d_rs, kpts_const))
        #print(get_shape(vertex3d_rs))
        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, get_shape(vertex3d_rs)[1]], tf.float32)])
        
        m_single_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        vertex2d = tf.matmul(m_single_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v]   = tf.split(axis=1, num_or_size_splits=2, value=vertex2d) #[vertex2d_u, vertex2d_v]   = tf.split(1, 2, vertex2d)
        vertex2d_u = vertex2d_u - 1 
        vertex2d_v = s - vertex2d_v

        landmarks_u.append(vertex2d_u)
        landmarks_v.append(vertex2d_v)

    return tf.stack(landmarks_u), tf.stack(landmarks_v)
'''

def rotate_shape(m, mshape, output_size = 224):

    n_size = get_shape(m)    
    n_size = n_size[0]

    m_single     = tf.split(axis = 0, num_or_size_splits = n_size, value = m)
    shape_single = tf.split(axis = 0, num_or_size_splits = n_size, value = mshape)
    
    vertex2ds = []

    for i in range(n_size):

        m_i = tf.transpose(tf.reshape(m_single[i], [4,2]))
        m_i_row1 = tf.nn.l2_normalize(m_i[0,0:3], dim = 0)
        m_i_row2 = tf.nn.l2_normalize(m_i[1,0:3], dim = 0)
        m_i_row3 = tf.concat([tf.reshape(tf.cross(m_i_row1, m_i_row2), shape = [1, 3]), tf.zeros([1, 1])], axis = 1)
                  
        m_i = tf.concat([m_i, m_i_row3], axis = 0)

        vertex3d_rs = tf.transpose(tf.reshape( shape_single[i], shape = [-1, 3] ))

        vertex4d = tf.concat(axis = 0, values = [vertex3d_rs, tf.ones([1, get_shape(vertex3d_rs)[1]], tf.float32)])
        
        vertex2d = tf.matmul(m_i, vertex4d, False, False)
        vertex2d = tf.transpose(vertex2d)
        
        [vertex2d_u, vertex2d_v, vertex2d_z]   = tf.split(axis=1, num_or_size_splits=3, value=vertex2d)
        vertex2d_u = vertex2d_u - 1
        vertex2d_v = output_size - vertex2d_v

        vertex2d = tf.concat(axis=1, values=[vertex2d_v, vertex2d_u, vertex2d_z])
        vertex2d = tf.transpose(vertex2d)

        vertex2ds.append(vertex2d)

    return tf.stack(vertex2ds)


def shading_old(L, normal):
    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743125
    c4 = 0.886227
    c5 = 0.247708

    L1, L2, L3 = tf.split(L, num_or_size_splits = 3, axis=1)

    L1 = tf.split(L1, num_or_size_splits = 9, axis=1)
    K1 = tf.stack([ [c1 * L1[8],  c1*L1[4], c1*L1[7], c2*L1[3] ], 
                    [c1 * L1[4], -c1*L1[8], c1*L1[5], c2*L1[1] ], 
                    [c1 * L1[7],  c1*L1[5], c3*L1[6], c2*L1[2] ],
                    [c2 * L1[3],  c2*L1[1], c2*L1[2], c4*L1[0] - c5*L1[6] ] ] )
    K1 = tf.transpose(K1, perm=[2, 3, 0, 1])

    L2 = tf.split(L2, num_or_size_splits = 9, axis=1)
    K2 = tf.stack([ [c1 * L2[8],  c1*L2[4], c1*L2[7], c2*L2[3] ], 
                    [c1 * L2[4], -c1*L2[8], c1*L2[5], c2*L2[1] ], 
                    [c1 * L2[7],  c1*L2[5], c3*L2[6], c2*L2[2] ],
                    [c2 * L2[3],  c2*L2[1], c2*L2[2], c4*L2[0] - c5*L2[6] ] ] )
    K2 = tf.transpose(K2, perm=[2, 3, 0, 1])

    L3 = tf.split(L3, num_or_size_splits = 9, axis=1)
    K3 = tf.stack([ [c1 * L3[8],  c1*L3[4], c1*L3[7], c2*L3[3] ], 
                    [c1 * L3[4], -c1*L3[8], c1*L3[5], c2*L3[1] ], 
                    [c1 * L3[7],  c1*L3[5], c3*L3[6], c2*L3[2] ],
                    [c2 * L3[3],  c2*L3[1], c2*L3[2], c4*L3[0] - c5*L3[6] ] ] )
    K3 = tf.transpose(K3, perm=[2, 3, 0, 1])


    shape = normal.get_shape().as_list()#[-1:]
    shape[-1] = 1
    ones_tensor = tf.ones(shape = shape)
    normal = tf.concat(values = [normal, ones_tensor], axis = 2)

    normal2 = tf.expand_dims(normal, axis = 2)

    A1 = tf.matmul(normal2, tf.tile(K1, [1, shape[1], 1, 1]))
    B1 = tf.matmul(A1, tf.matrix_transpose(normal2))
    B1 = tf.reshape(B1, shape = [shape[0], shape[1], 1])

    A2 = tf.matmul(normal2, tf.tile(K1, [1, shape[1], 1, 1]))
    B2 = tf.matmul(A2, tf.matrix_transpose(normal2))
    B2 = tf.reshape(B2, shape = [shape[0], shape[1], 1])

    A3 = tf.matmul(normal2, tf.tile(K1, [1, shape[1], 1, 1]))
    B3 = tf.matmul(A3, tf.matrix_transpose(normal2))
    B3 = tf.reshape(B3, shape = [shape[0], shape[1], 1])

    return tf.concat([B1, B2, B3], axis = 2)


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=True, name=name)


# Def custom square function using np.square instead of tf.square:
def ZBuffer_Rendering_op(s2d, vertex, tri):
   
    with ops.name_scope("ZBuffer_Rendering", [s2d, vertex]) as name:
        sqr_x = py_func(ZBuffer_Rendering,
                        [s2d, vertex, tri],
                        [tf.float32],
                        name=name,
                        grad=_ZBuffer_Rendering_Grad)  # <-- here's the call to the gradient
        return sqr_x[0]

# Actual gradient:
def _ZBuffer_Rendering_Grad(op, grad):
    x = op.inputs[0]
    y = op.inputs[1]
    z = op.inputs[2]

    print(op.inputs)
    return 0 * x , 0 * y, 0*z       
    

def ZBuffer_Rendering(s2d, vertex, tri):

    TRI_NUM = tri.shape[1] - 1

    width = 96
    height = 96   

    point1 = s2d[:, tri[0,:-1]]
    point2 = s2d[:, tri[1,:-1]]
    point3 = s2d[:, tri[2,:-1]]

    cent3d = (vertex[:, tri[0, :-1]] + vertex[:, tri[1, :-1]] + vertex[:, tri[2, :-1]]) / 3

    r = np.sum(np.square(cent3d), axis=0)

    imgr = np.zeros([height, width])
    img = (TRI_NUM)*np.ones([height, width], dtype=np.float32)

    for i in range (TRI_NUM):
        pt1 = point1[:,i]
        pt2 = point2[:,i]
        pt3 = point3[:,i]

        umin = int( np.ceil(min([pt1[0], pt2[0], pt3[0]]))   )
        umax = int( np.floor(max([pt1[0], pt2[0], pt3[0]]))  )

        vmin = int( np.ceil(min([pt1[1], pt2[1], pt3[1]]))   )
        vmax = int( np.ceil(max([pt1[1], pt2[1], pt3[1]]))   )
        if (umax < width and vmax < height and umin >= 0 and vmin >= 0 ):
            for u in range (umin, umax+1):
                for v in range (vmin, vmax+1):
                    if (imgr[u,v] < r[i]  and triCpoint(np.asarray([u, v]), pt1, pt2, pt3)):
                        imgr[u,v] = r[i]
                        img[u,v] = i

    return img.astype(np.float32)



def triCpoint(point, pt1, pt2, pt3):
    v0 = pt3 - pt1; #C - A
    v1 = pt2 - pt1; #B - A
    v2 = point - pt1;
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inverDeno = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6)
    u = (dot11 * dot02 - dot01 * dot12) * inverDeno
    if (u < 0 or u > 1):
        return 0

    v = (dot00 * dot12 - dot01 * dot02) * inverDeno
    if (v < 0 or v > 1):
        return 0

    return u + v <= 1
            

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    #print('x.get_shape()')
    #print(x.get_shape())

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    #print('b.get_shape()')
    #print(b.get_shape())

    indices = tf.stack([b, y, x], 3)

    #print('indices.get_shape()')
    #print(indices.get_shape())

    return tf.gather_nd(img, indices)



def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the 
    normalized coordinates provided by the sampling grid. Note that 
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    #print('img.get_shape()')
    #print(img.get_shape())

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    #x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    #y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)
    
    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out


def splineWarping(image, landmark_old_u, landmark_old_v, landmark_new_u, landmark_new_v, output_size):
    n_size = get_shape(image)    
    n_size = n_size[0]

    landmark_old_u = tf.split(landmark_old_u, n_size)
    landmark_old_v = tf.split(landmark_old_v, n_size)
    landmark_new_u = tf.split(landmark_new_u, n_size)
    landmark_new_v = tf.split(landmark_new_v, n_size)

    indicies = np.zeros([output_size*output_size,2])
    for i in range(output_size):
        for j in range(output_size):
            indicies[i*output_size+j ] = [i,j]

    indicies_const = tf.constant(indicies, tf.float32)
    [indicies_const_v, indicies_const_u] = tf.split(axis=1, num_or_size_splits=2, value=indicies_const)


    pixel_u = []
    pixel_v = []
    for i in range(n_size):

        landmark_old_u_i = tf.squeeze(landmark_old_u[i], axis = 0)
        landmark_old_v_i = tf.squeeze(landmark_old_v[i], axis = 0)
        landmark_new_u_i = tf.squeeze(landmark_new_u[i], axis = 0)
        landmark_new_v_i = tf.squeeze(landmark_new_v[i], axis = 0)

        vertex2d = tf.concat(axis=1, values=[landmark_old_v_i, landmark_old_u_i])


        NN = get_shape(landmark_old_u_i)[0]

        #print('get_shape(vertex2d)')
        #print(get_shape(vertex2d))

        ## Warping
        # Compute coeifficient
        distance =  tf.square(tf.tile(tf.reshape(vertex2d, shape = [1,NN, 2]), [NN, 1, 1]) - tf.tile(tf.reshape(vertex2d, shape = [NN,1, 2]), [1, NN, 1]))
        distance = tf.sqrt(tf.reduce_sum(distance, axis = 2))

        A = distance
        B = tf.concat(axis=1, values=[tf.ones([vertex2d.get_shape()[0], 1], tf.float32), vertex2d ])

        LHS =  tf.concat(axis=0, values=[ tf.concat(axis=1, values=[A, B]),  tf.concat(axis=1, values=[tf.transpose(B), tf.zeros([3,3])])  ])
        #print("LHS")
        #print(get_shape(LHS))

        G_u = -landmark_new_u_i + landmark_old_u_i
        G_v = -landmark_new_v_i + landmark_old_v_i

        #print('get_shape(G_u)')
        #print(get_shape(G_u))

        #print('get_shape(landmark_new_u_i)')
        #print(get_shape(landmark_new_u_i))

        
        matrix_v = tf.matrix_solve(LHS, rhs = tf.concat(axis=0, values=[G_v, tf.zeros([3,1])]) )
        matrix_u = tf.matrix_solve(LHS, rhs = tf.concat(axis=0, values=[G_u, tf.zeros([3,1])]) )


        # Warping
        pixel_u_i = tf.reshape(indicies_const_u + splineInterpolation(indicies_const, vertex2d, matrix_u), [output_size, output_size])
        pixel_v_i = tf.reshape(indicies_const_v + splineInterpolation(indicies_const, vertex2d, matrix_v), [output_size, output_size])

        pixel_u.append(pixel_u_i)
        pixel_v.append(pixel_v_i)

             
    return bilinear_sampler(image, pixel_u, pixel_v)




def splineInterpolation(x, x1, matrix):

    N  = int(x.get_shape()[0])
    N1 = int(x1.get_shape()[0])

    distance = tf.square(tf.tile(tf.reshape(x, shape = [N,1, 2]), [1, N1, 1]) - tf.tile(tf.reshape(x1, shape = [1,N1, 2]), [N, 1, 1]))
    distance = tf.sqrt(tf.reduce_sum(distance, axis = 2))

    A = distance
    B = tf.concat(axis=1, values=[tf.ones([x.get_shape()[0], 1], tf.float32), x ])

    return tf.matmul(tf.concat(axis=1, values=[A, B]), matrix)