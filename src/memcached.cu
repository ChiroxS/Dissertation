#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>

#include "gpu_hash_functions.cu"
#include "cpu_buffer.h"
#include "slab.h"

void print_buffer(elem_buffer_t** cpu_buffer);
void fill_insert_buffer(FILE** file_pointer, elem_buffer_t** cpu_buffer, slab_block* slab[]);
void transfer_data(elem_buffer_t** host_buffer, elem_buffer_t** device_buffer, direction_t dir);

void insert_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table);

int  get_free_slab(slab_block_t* slab[]);


int main()
{    
    FILE* key_value_file_pointer;
    key_value_file_pointer = fopen("data/data.txt", "r");
    if(key_value_file_pointer == NULL) {
        printf("File path NULL \n");
    } else {
        printf("File path OK \n");
    }
    
    //******************************************************************************
    // Initialize memory
    bucket_t *hash_table;
    cudaMallocManaged(&hash_table, HASH_NO_BUCKETS * sizeof(bucket_t));
    cudaMemset     (hash_table, 0, HASH_NO_BUCKETS * sizeof(bucket_t));

    elem_buffer_t *cpu_buffer;
    cpu_buffer = (elem_buffer_t*)malloc(sizeof(elem_buffer_t));
    cpu_buffer->key_search = (key_search_t*)malloc(MAX_SEARCH_JOBS * sizeof(key_search_t));
    cpu_buffer->key_insert = (key_insert_t*)malloc(MAX_INSERT_JOBS * sizeof(key_insert_t));
    cpu_buffer->key_delete = (key_delete_t*)malloc(MAX_DELETE_JOBS * sizeof(key_delete_t));
    cpu_buffer->locations  =   (location_t*)malloc(MAX_SEARCH_JOBS * sizeof(location_t  ));

    elem_buffer_t *gpu_buffer;
    cudaMallocManaged(&gpu_buffer, sizeof(elem_buffer_t));
    cudaMallocManaged(&gpu_buffer->key_search, MAX_SEARCH_JOBS * sizeof(key_search_t));
    cudaMallocManaged(&gpu_buffer->key_insert, MAX_INSERT_JOBS * sizeof(key_insert_t));
    cudaMallocManaged(&gpu_buffer->key_delete, MAX_DELETE_JOBS * sizeof(key_delete_t));
    cudaMallocManaged(&gpu_buffer->locations , MAX_SEARCH_JOBS * sizeof(location_t  ));
    
    slab_block *slab; 
    slab = (slab_block*)malloc(SLAB_LEN * sizeof(slab_block));
    //******************************************************************************
    
    fill_insert_buffer(&key_value_file_pointer, &cpu_buffer, &slab);
    //print_buffer(&cpu_buffer);
    transfer_data(&cpu_buffer, &gpu_buffer, HOST_TO_DEVICE);
    insert_gpu(&gpu_buffer, &hash_table);


    /*
    fill_serch_buffer(cpu_buffer);
    transfer_data(cpu_buffer, gpu_buffer, HOST_TO_DEVICE);
    search(); 
    */ 

    //******************************************************************************
    fclose(key_value_file_pointer);
    // Free memory
    cudaFree(hash_table);
    cudaFree(gpu_buffer->key_search);
    cudaFree(gpu_buffer->key_insert);
    cudaFree(gpu_buffer->key_delete);
    cudaFree(gpu_buffer->locations);
    cudaFree(gpu_buffer);

    free(cpu_buffer->key_search);
    free(cpu_buffer->key_insert);
    free(cpu_buffer->key_delete);
    free(cpu_buffer->locations);
    free(cpu_buffer);

    free(slab);
    //******************************************************************************
}

void insert_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table) { 
    gpu_insert_key <<< 1, 8 >>> ( (*gpu_buffer)->key_insert , (*hash_table), 1);
    cudaDeviceSynchronize();
}

void transfer_data(elem_buffer_t** host_buffer, elem_buffer_t** device_buffer, direction_t dir) { 
    if(dir == HOST_TO_DEVICE) { 
        
        /*
        cudaMemcpy(device_buffer->nr_search_keys, host_buffer->nr_search_keys, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_buffer->nr_insert_keys, host_buffer->nr_insert_keys, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(device_buffer->nr_delete_keys, host_buffer->nr_delete_keys, sizeof(uint32_t), cudaMemcpyHostToDevice);
         */ 

        cudaMemcpy((*device_buffer)->key_search, (*host_buffer)->key_search,  (*host_buffer)->nr_search_keys * sizeof(key_search_t), cudaMemcpyHostToDevice);
        cudaMemcpy((*device_buffer)->key_insert, (*host_buffer)->key_insert,  (*host_buffer)->nr_insert_keys * sizeof(key_insert_t), cudaMemcpyHostToDevice);
        cudaMemcpy((*device_buffer)->key_delete, (*host_buffer)->key_delete,  (*host_buffer)->nr_delete_keys * sizeof(key_delete_t), cudaMemcpyHostToDevice);
    } 

    if(dir == DEVICE_TO_HOST) { 
        cudaMemcpy((*host_buffer)->locations, (*device_buffer)->locations, (*host_buffer)->nr_search_keys * (sizeof(location_t)), cudaMemcpyDeviceToHost);
    } 
}

void fill_insert_buffer(FILE** file_pointer, elem_buffer_t** cpu_buffer, slab_block* slab[]) { 
    char key_value_buffer[FILE_BUFFER_LEN];
    int slab_index = get_free_slab(slab);
    for(int i = 0; i < MAX_INSERT_JOBS; i++) {
        //***************************************************************************
        // copy key to slab
        fgets(key_value_buffer, FILE_BUFFER_LEN, *file_pointer);
        //printf("%s", key_value_buffer);
        strncpy((*slab)[slab_index].item[i].key, key_value_buffer, KEY_LEN);
        //***************************************************************************
        uint64_t hash_result; 
        hash_result = *(uint64_t*) (&key_value_buffer[0]); // evil char to uint64 bit level hacking
        for(int j = 8; j <= KEY_LEN - 8; j+= 8) {
            hash_result = hash_result ^ *(uint64_t*) (&key_value_buffer[j]);
        }
        //printf("Hash result for key %sis 0x%I64x\n", key_value_buffer, hash_result);
        (*cpu_buffer)->key_insert[i].hash      = (hash_t)       (hash_result >> 32);
        (*cpu_buffer)->key_insert[i].signature = (signature_t)  (hash_result      );
        (*cpu_buffer)->key_insert[i].location = (i & SLAB_ITEM_MASK) | (slab_index << NR_SLAB_ITEMS_LOG) ;
        //***************************************************************************
        // copy value
        fgets(key_value_buffer, FILE_BUFFER_LEN, *file_pointer);
        //printf("%s", key_value_buffer);
        strncpy((*slab)[slab_index].item[i].value, key_value_buffer, VALUE_LEN);
        //***************************************************************************
    }
    (*cpu_buffer)->nr_insert_keys = MAX_INSERT_JOBS;
    (*slab)[slab_index].occupied = 1;
}

int get_free_slab(slab_block* slab[]) {
    int i = 0;
    while(i < SLAB_LEN) {
        if((*slab)[i].occupied == 0) {
            //printf("Found empty slab at index %d \n", i);
            break;
        } else
            i++; 
    }
    return i;
}


void print_buffer(elem_buffer_t** cpu_buffer) { 
    printf("**************\n");
    printf("Printing buffer\n");
    printf("**************\n");
    printf("Insert jobs\n");
    for(int i = 0; i < (*cpu_buffer)->nr_insert_keys;i++) {
        printf("hash: 0x%x, signature: 0x%x, location: 0x%x \n",(*cpu_buffer)->key_insert[i].hash, (*cpu_buffer)->key_insert[i].signature, (*cpu_buffer)->key_insert[i].location );
    }
    printf("Search jobs\n");
    for(int i = 0; i < (*cpu_buffer)->nr_search_keys;i++) {
        printf("hash: 0x%x, signature: 0x%x \n",(*cpu_buffer)->key_search[i].hash, (*cpu_buffer)->key_search[i].signature );
    }
    printf("Delete jobs\n");
    for(int i = 0; i < (*cpu_buffer)->nr_delete_keys;i++) {
        printf("hash: 0x%x, signature: 0x%x \n",(*cpu_buffer)->key_delete[i].hash, (*cpu_buffer)->key_delete[i].signature  );
    }

    printf("Locations\n");
    for(int i = 0; i < (*cpu_buffer)->nr_search_keys;i++) {
        printf("0x%x",(*cpu_buffer)->locations[i] );
    }

}