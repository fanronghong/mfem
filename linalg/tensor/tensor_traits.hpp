// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_TENSOR_TRAITS
#define MFEM_TENSOR_TRAITS

#include "containers/container_traits.hpp"
#include "containers/static_container.hpp"
#include "layouts/layout_traits.hpp"

namespace mfem
{

/// Forward declaration of the Tensor class
template <typename Container, typename Layout>
class Tensor;

/**
 * Tensor Traits:
 *    Traits can be seen as compilation time functions operating on types.
 * There is two types of traits, traits returning values, and traits returning
 * types. The following traits allow to analyze tensors at compilation.
*/

/// get_tensor_rank
/** Trait to get the rank of a tensor at compilation.
    ex: `constexpr int Rank = get_tensor_rank<Tensor>;'
*/
template <typename Tensor>
struct get_tensor_rank_v
{
   static constexpr int value = Error;
};

template <typename Container, typename Layout>
struct get_tensor_rank_v<Tensor<Container,Layout>>
{
   static constexpr int value = get_layout_rank<Layout>;
};

template <typename Tensor>
constexpr int get_tensor_rank = get_tensor_rank_v<Tensor>::value;

/// get_tensor_value_type
/** Return the type of values stored by the Tensor's container.
    ex: `using T = get_tensor_value_type<Tensor>;'
*/
template <typename Tensor>
struct get_tensor_value_type_t;

template <typename C, typename L>
struct get_tensor_value_type_t<Tensor<C,L>>
{
   using type = get_container_type<C>;
};

template <typename C, typename L>
struct get_tensor_value_type_t<const Tensor<C,L>>
{
   using type = get_container_type<C>;
};

template <typename Tensor>
using get_tensor_value_type = typename get_tensor_value_type_t<Tensor>::type;

template <typename Tensor>
using get_tensor_type = typename get_tensor_value_type_t<Tensor>::type;

/// is_dynamic_tensor
/** Return true if the tensor's layout is dynamically sized.
    ex: `constexpr bool is_dynamic = is_dynamic_tensor<Tensor>;'
*/
template <typename Tensor>
struct is_dynamic_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_dynamic_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>;
};

template <typename C, typename L>
struct is_dynamic_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_dynamic_layout<L>;
};

template <typename Tensor>
constexpr bool is_dynamic_tensor = is_dynamic_tensor_v<Tensor>::value;

/// is_static_tensor
/** Return true if the tensor's layout is statically sized.
*/
template <typename Tensor>
struct is_static_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_static_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_static_layout<L>;
};

template <typename C, typename L>
struct is_static_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_static_layout<L>;
};

template <typename Tensor>
constexpr bool is_static_tensor = is_static_tensor_v<Tensor>::value;

/// is_serial_tensor
/** Return true if the tensor's is not distributed over threads.
*/
template <typename Tensor>
struct is_serial_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_serial_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_serial_layout<L>;
};

template <typename C, typename L>
struct is_serial_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_serial_layout<L>;
};

template <typename Tensor>
constexpr bool is_serial_tensor = is_serial_tensor_v<Tensor>::value;

/// is_2d_threaded_tensor
/** Return true if the tensor's layout is 2d threaded.
*/
template <typename Tensor>
struct is_2d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_2d_threaded_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>;
};

template <typename C, typename L>
struct is_2d_threaded_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_2d_threaded_layout<L>;
};

template <typename Tensor>
constexpr bool is_2d_threaded_tensor = is_2d_threaded_tensor_v<Tensor>::value;

/// is_3d_threaded_tensor
/** Return true if the tensor's layout is 3d threaded.
*/
template <typename Tensor>
struct is_3d_threaded_tensor_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct is_3d_threaded_tensor_v<Tensor<C,L>>
{
   static constexpr bool value = is_3d_threaded_layout<L>;
};

template <typename C, typename L>
struct is_3d_threaded_tensor_v<const Tensor<C,L>>
{
   static constexpr bool value = is_3d_threaded_layout<L>;
};

template <typename Tensor>
constexpr bool is_3d_threaded_tensor = is_3d_threaded_tensor_v<Tensor>::value;

/// is_serial_tensor_dim
/** Return true if the tensor's layout dimension N is not threaded.
*/
template <typename Tensor, int N>
constexpr bool is_serial_tensor_dim = is_serial_layout_dim<
                                         typename Tensor::layout,
                                         N
                                      >;

/// is_threaded_tensor_dim
/** Return true if the tensor's layout dimension N is threaded.
*/
template <typename Tensor, int N>
constexpr bool is_threaded_tensor_dim = is_threaded_layout_dim<
                                           typename Tensor::layout,
                                           N
                                        >;

/// get_tensor_size
/** Return the compilation time size of the dimension N, returns Dynamic for
    dynamic sizes.
*/
template <int N, typename Tensor>
struct get_tensor_size_v;
// {
//    static constexpr int value = Error;
// };

template <int N, typename C, typename L>
struct get_tensor_size_v<N, Tensor<C,L>>
{
   static constexpr int value = get_layout_size<N, L>;
};

template <int N, typename C, typename L>
struct get_tensor_size_v<N, const Tensor<C,L>>
{
   static constexpr int value = get_layout_size<N, L>;
};

template <int N, typename Tensor>
constexpr int get_tensor_size = get_tensor_size_v<N, Tensor>::value;

/// get_tensor_batch_size
/** Return the tensor's batchsize, the batchsize being the number of elements
    treated per block of threads.
*/
template <typename Tensor>
struct get_tensor_batch_size_v
{
   static constexpr int value = Error;
};

template <typename C, typename L>
struct get_tensor_batch_size_v<Tensor<C,L>>
{
   static constexpr int value = get_layout_batch_size<L>;
};

template <typename C, typename L>
struct get_tensor_batch_size_v<const Tensor<C,L>>
{
   static constexpr int value = get_layout_batch_size<L>;
};

template <typename Tensor>
constexpr int get_tensor_batch_size = get_tensor_batch_size_v<Tensor>::value;

/// get_tensor_capacity
/** Return the number of values stored per thread.
*/
template <typename Tensor>
struct get_tensor_capacity_v
{
   static constexpr int value = Error;
};

template <typename C, typename L>
struct get_tensor_capacity_v<Tensor<C,L>>
{
   static constexpr int value = get_layout_capacity<L>;
};

template <typename C, typename L>
struct get_tensor_capacity_v<const Tensor<C,L>>
{
   static constexpr int value = get_layout_capacity<L>;
};

template <typename Tensor>
constexpr int get_tensor_capacity = get_tensor_capacity_v<Tensor>::value;

/// has_pointer_container
/** Return true if the tensor's container is a pointer type.
*/
template <typename Tensor>
struct has_pointer_container_v
{
   static constexpr bool value = false;
};

template <typename C, typename L>
struct has_pointer_container_v<Tensor<C,L>>
{
   static constexpr bool value = is_pointer_container<C>;
};

template <typename C, typename L>
struct has_pointer_container_v<const Tensor<C,L>>
{
   static constexpr bool value = is_pointer_container<C>;
};

template <typename Tensor>
constexpr bool has_pointer_container = has_pointer_container_v<Tensor>::value;

/// is_static_matrix
/** Return true if the tensor is a statically sized matrix.
*/
template <int N, int M, typename Tensor>
struct is_static_matrix_v
{
   static constexpr bool value = is_static_tensor<Tensor> &&
                                 get_tensor_rank<Tensor> == 2 &&
                                 get_tensor_size<0,Tensor> == N &&
                                 get_tensor_size<1,Tensor> == M;
};

template <int N, int M, typename Tensor>
constexpr bool is_static_matrix = is_static_matrix_v<N,M,Tensor>::value;

/// is_dynamic_matrix
/** Return true if the tensor is a dynamically sized matrix.
*/
template <typename Tensor>
struct is_dynamic_matrix_v
{
   static constexpr bool value = is_dynamic_tensor<Tensor> &&
                                 get_tensor_rank<Tensor> == 2;
};

template <typename Tensor>
constexpr bool is_dynamic_matrix = is_dynamic_matrix_v<Tensor>::value;


/// get_tensor_result_type
/** Return a tensor type with static sizing (even for dynamic layouts)
    compatible with the result of an operator on the Tensor type.
    ex:
    ```
    using Tensor = DynamicTensor<4>;
    StaticResultTensor<Tensor,Dynamic,Dynamic> u;// the type is DynamicTensor<2>
    ```
    Allows to write algorithms which are agnostic of the input type Tensor.
*/
template <typename Tensor, typename Enable = void>
struct get_tensor_result_type;

template <typename MyTensor>
struct get_tensor_result_type<MyTensor,
                              std::enable_if_t<
                                 is_static_tensor<MyTensor>>>
{
   using T = typename MyTensor::T;

   template <int... Dims>
   using Layout = typename get_layout_result_type<typename MyTensor::layout>
                     ::template type<Dims...>;

   template <int... Dims>
   using Container = StaticContainer<T,get_layout_capacity<Layout<Dims...>>>;

   template <int... Dims>
   using type = Tensor<Container<Dims...>, Layout<Dims...>>;

   template <int... Dims>
   using static_type = type<Dims...>;
};

template <typename MyTensor>
struct get_tensor_result_type<MyTensor,
                              std::enable_if_t<
                                 is_dynamic_tensor<MyTensor>>>
{
   using T = typename MyTensor::T;

   template <int Rank>
   using Layout = typename get_layout_result_type<typename MyTensor::layout>
                     ::template type<Rank>;

   template <int Rank>
   using Container = StaticContainer<T, get_layout_capacity< Layout<Rank> > >;

   template <int Rank>
   using type = Tensor<Container<Rank>, Layout<Rank>>;

   template <template <int> class Tensor, int... Sizes>
   using static_tensor_wrap = Tensor< sizeof...(Sizes) >;

   template <int... Dims>
   using static_type = static_tensor_wrap<type,Dims...>;
};

template <typename Tensor, int... Dims>
using StaticResultTensor = typename get_tensor_result_type<Tensor>
                              ::template static_type<Dims...>;

} // namespace mfem

#endif // MFEM_TENSOR_TRAITS
