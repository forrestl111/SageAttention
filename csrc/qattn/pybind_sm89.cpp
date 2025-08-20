/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_cuda_sm89.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f8_accum_f32_attn", &qk_int8_sv_f8_accum_f32_attn, "QK int8 sv f8 accum f32 attn",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("tensor_layout"),
        pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"), pybind11::arg("sm_scale"),
        pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
        
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn", &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn, "QK int8 sv f8 accum f32 fuse v scale attn",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("value_scale"),
        pybind11::arg("tensor_layout"), pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"),
        pybind11::arg("sm_scale"), pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
        
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn", &qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn, "QK int8 sv f8 accum f32 fuse v scale fuse v mean attn",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("value_scale"),
        pybind11::arg("value_mean"), pybind11::arg("tensor_layout"), pybind11::arg("is_causal"),
        pybind11::arg("qk_quant_gran"), pybind11::arg("sm_scale"), pybind11::arg("return_lse"),
        pybind11::arg("block_index") = pybind11::none());

  m.def("qk_int8_sv_f8_accum_f32_attn_inst_buf", &qk_int8_sv_f8_accum_f32_attn_inst_buf, "QK int8 sv f8 accum f32 attn inst buf",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("tensor_layout"),
        pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"), pybind11::arg("sm_scale"),
        pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
        
  m.def("qk_int8_sv_f8_accum_f16_attn_inst_buf", &qk_int8_sv_f8_accum_f16_attn_inst_buf, "QK int8 sv f8 accum f16 attn inst buf",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("tensor_layout"),
        pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"), pybind11::arg("sm_scale"),
        pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
        
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf, "QK int8 sv f8 accum f32 fuse v scale attn inst buf",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("value_scale"),
        pybind11::arg("tensor_layout"), pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"),
        pybind11::arg("sm_scale"), pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
        
  m.def("qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf", &qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf, "QK int8 sv f8 accum f16 fuse v scale attn inst buf",
        pybind11::arg("query"), pybind11::arg("key"), pybind11::arg("value"), pybind11::arg("output"),
        pybind11::arg("query_scale"), pybind11::arg("key_scale"), pybind11::arg("value_scale"),
        pybind11::arg("tensor_layout"), pybind11::arg("is_causal"), pybind11::arg("qk_quant_gran"),
        pybind11::arg("sm_scale"), pybind11::arg("return_lse"), pybind11::arg("block_index") = pybind11::none());
}