/**
 * Заголовки CUDA 12.0 из Ubuntu apt не содержат символов >= 12.4,
 * которые использует XGBoost 3.0. Драйвер 570+ поддерживает их в рантайме.
 */
#pragma once

#include <cuda.h>

#if defined(CUDA_VERSION) && CUDA_VERSION < 12040
#ifndef CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
#define CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID static_cast<CUdevice_attribute>(134)
#endif
#ifndef CU_MEM_LOCATION_TYPE_HOST_NUMA
#define CU_MEM_LOCATION_TYPE_HOST_NUMA static_cast<CUmemLocationType>(0x2)
#endif
#endif
