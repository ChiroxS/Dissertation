
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include<iostream>
#include<string>
#include <cstdlib>
#include <chrono>

#include "gpu_hash_functions.cu"

int main()
{
    int N = 17;
    int hash_values[17]      = {0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA, 0xAAAAAAAA};
    int signature_values[17] = {0xB1B1, 0xB2B2, 0xB3B3, 0xB4B4, 0xB5B5, 0xB6B6, 0xB711, 0xB811, 0xC111, 0xC211, 0xC311, 0xC411, 0xC511, 0xC611, 0xC711, 0xC811, 0xD111};
    int location_values[17]  = {0x0111, 0x0211, 0x0311, 0x0411, 0x0511, 0x0611, 0x0711, 0x0811, 0x0911, 0x0A11, 0x0B11, 0x0C11, 0x0D11, 0x0E11, 0x0F11, 0x1011, 0x1111};

    bucket_t *hash_table;
    cudaMallocManaged(&hash_table, HASH_NO_BUCKETS * sizeof(bucket_t));
    cudaMemset(hash_table, 0, HASH_NO_BUCKETS * sizeof(bucket_t));
    
    key_search_t *key_search_device;
    key_insert_t *key_insert_device;

    key_search_t *key_search_host;
    key_insert_t *key_insert_host;

    location_t *loc_host;
    location_t *loc_device;

    key_search_host = (key_search_t*)malloc(N*sizeof(key_search_t));
    key_insert_host = (key_insert_t*)malloc(sizeof(key_insert_t));

    loc_host        = (location_t*)malloc(N*sizeof(location_t));
 
    cudaMallocManaged(&key_insert_device, sizeof(key_insert_t));
	cudaMallocManaged(&key_search_device, N * sizeof(key_search_t));
    cudaMallocManaged(&loc_device  , N * sizeof(location_t));


    ///////////////////// INSERT
    for(int i = 0; i < N; i++) {
        key_insert_host[0].hash = hash_values[0];
        key_insert_host[0].signature = signature_values[i];
        // uncomment below to insert to second bucket
        //key_insert_host[0].hash = key_insert_host[0].hash ^ key_insert_host[0].signature;
        key_insert_host[0].location = location_values[i];
        cudaMemcpy(key_insert_device, key_insert_host,  sizeof(key_insert_t), cudaMemcpyHostToDevice);
        gpu_insert_key <<< 1, 8 >>> (key_insert_device, hash_table, 1);
        cudaDeviceSynchronize();
    }
    ////////////////////

    //////////////////// SEARCH
    for(int i = 0; i < N; i++) {
        key_search_host[i].hash = hash_values[0];
        key_search_host[i].signature = signature_values[i];
    }

    cudaMemcpy(key_search_device, key_search_host, N * sizeof(key_search_t), cudaMemcpyHostToDevice);
    gpu_search_key <<< 1, N * 8 >>> (key_search_device, loc_device, hash_table, 1);
    cudaDeviceSynchronize();
    ////////////////////

    cudaMemcpy(loc_host, loc_device, N * (sizeof(location_t)), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++) {
        std::cout << std::hex << loc_host[i] << std::endl; 
    }
    cudaFree(hash_table);
	cudaFree(key_insert_device);
    cudaFree(key_search_device);
    cudaFree(key_search_device);

	free(key_search_host);
	free(key_insert_host);
	free(loc_host);

    return 0;
}
