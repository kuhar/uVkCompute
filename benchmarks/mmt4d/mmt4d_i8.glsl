// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#version 450 core
#pragma use_vulkan_memory_model

#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require

layout(binding = 0) buffer InputA { i8vec4 x[]; } inputA;
layout(binding = 1) buffer InputB { i8vec4 x[]; } inputB;
layout(binding = 2) buffer Output { int32_t x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint VECTORIZE_K = 4;
const uint K_VEC = K / VECTORIZE_K;
const uint K0_VEC = K0 / VECTORIZE_K;

const uint strideA = K_VEC; // Stride of the `inputA` matrix.
const uint strideB = K_VEC; // Stride of the `inputB` matrix.
const uint strideC = N; // Stride of the `outputO` matrix.

// Each workgroup processes an output tile of size [M0 x N0], therefore
// each thread processes a [M0/WG_Y x N0/WG_X] subview.
const uint C_ROWS = M0 / WG_Y;
const uint C_COLS = N0 / WG_X;

/// Returns the index of `X[i, j]`, where `X` is a 2D matrix of stride |stride|.
uint coordToOffset(uint i, uint j, uint stride) { return stride * i + j; }

void main() {
  uvec2 gID = gl_WorkGroupID.xy;
  uvec2 laneId = gl_LocalInvocationID.xy;

  // The start offsets of the tile processed by this thread in this workgroup.
  uint x_offset = gID.x * N0 + laneId.x;
  uint y_offset = gID.y * M0 + laneId.y;

  i8vec4 RHS[K0_VEC][C_COLS];
  int32_t C[C_ROWS][C_COLS]; // Local data for the output.

  // Initialize result to zero.
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      C[i][j] = 0;
    }
  }

  for (uint k = 0; k < K_VEC; k += K0_VEC) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      uint x = x_offset + j * WG_X;
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        RHS[kk][j] = inputB.x[coordToOffset(x, k + kk, strideB)];
      }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
      [[unroll]] for (uint kk = 0; kk < K0_VEC; ++kk) {
        uint y = y_offset + i * WG_Y;
        i8vec4 lhs = inputA.x[coordToOffset(y, k + kk, strideA)];
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          uint x = x_offset + j * WG_X;

          // Calculate the inner product `C[i, j] := sum(A[i, ..] * B[j, ..])`.
          i16vec4 mul = i16vec4(lhs) * i16vec4(RHS[kk][j]);
          C[i][j] += mul.x + mul.y + mul.z + mul.w;
        }
      }
    }
  }

  // Store the accumulated results in `outputO`.
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    uint y = gID.y * M0 + laneId.y + i * WG_Y;
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      uint x = gID.x * N0 + laneId.x + j * WG_X;
      outputO.x[coordToOffset(y, x, strideC)] = C[i][j];
    }
  }
}
