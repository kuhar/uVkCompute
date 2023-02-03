# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-------------------------------------------------------------------------------
# mmt4d (Adreno)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    mmt4d_i8_shader_adreno
  SRC
    "mmt4d_i8.glsl"
  PERMUTATION
    "M0=[4|8|16|32]"
    "N0=[128|256]"
    "K0=[4|8|16]"
    "WG_X=32"
    "WG_Y=2"
)

uvkc_cc_binary(
  NAME
    mmt4d_adreno
  SRCS
    "mmt4d_main.cc"
  DEPS
    ::mmt4d_i8_shader_adreno
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_ADRENO
)

#-------------------------------------------------------------------------------
# mmt4d (Mali Valhall)
#-------------------------------------------------------------------------------

uvkc_glsl_shader_permutation(
  NAME
    mmt4d_i8_shader_valhall
  SRC
    "mmt4d_i8.glsl"
  PERMUTATION
    "M0=[4|8|16|32]"
    "N0=[64|128]"
    "K0=[4|8|16]"
    "WG_X=16"
    "WG_Y=1"
)

uvkc_cc_binary(
  NAME
    mmt4d_mali_valhall
  SRCS
    "mmt4d_main.cc"
  DEPS
    ::mmt4d_i8_shader_valhall
    benchmark::benchmark
    uvkc::benchmark::core
    uvkc::benchmark::main
  COPTS
    -DUVKC_MALI_VALHALL
)
