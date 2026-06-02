/**
 * Copyright 2024, XGBoost contributors
 */
#include "cuda_rt_utils.h"  // for RtVersion
#include "cuda_old_toolkit_compat.h"
#include "device_helpers.cuh"
#include "xgboost/windefs.h"  // for xgboost_IS_WIN

namespace dh {
PinnedMemory::PinnedMemory() {
#if defined(xgboost_IS_WIN)
  this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
#else
  // CUDA toolkit 12.0 из apt: заголовки без HOST_NUMA; virtual mem на unmap падает.
  // Используем pinned memory, GPU-обучение работает стабильно.
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
  std::int32_t major{0}, minor{0};
  xgboost::curt::DrVersion(&major, &minor);
  if (major >= 12 && minor >= 5) {
    this->impl_.emplace<detail::GrowOnlyVirtualMemVec>(CU_MEM_LOCATION_TYPE_HOST_NUMA);
  } else {
    this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
  }
#else
  this->impl_.emplace<detail::GrowOnlyPinnedMemoryImpl>();
#endif
#endif
}
}  // namespace dh
