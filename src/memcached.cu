#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gpu_hash_functions.cu"
#include "cpu_buffer.h"
#include "slab.h"

void print_buffer(elem_buffer_t** cpu_buffer);
void fill_insert_buffer(FILE** file_pointer, elem_buffer_t** cpu_buffer, slab_block* slab[]);
void fill_search_buffer(FILE** file_pointer, elem_buffer_t** cpu_buffer);
void transfer_data(elem_buffer_t** host_buffer, elem_buffer_t** device_buffer, direction_t dir);

void insert_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table);
void search_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table);

int  get_free_slab(slab_block_t* slab[]);
void write_search_data_to_file(elem_buffer_t** cpu_buffer, FILE** file_pointer,  slab_block_t* slab[]);

int main()
{    
    cudaError_t cudaStatus;
    //******************************************************************************
    // External file pointers 
    FILE* key_value_file_pointer;
    key_value_file_pointer = fopen("data/data_write.txt", "r");
    if(key_value_file_pointer == NULL) {
        printf("Key-value file path NULL \n");
    } else {
        printf("Key-value file path OK \n");
    }
    
    FILE* key_file_pointer;
    key_file_pointer = fopen("data/data_read.txt", "r");
    if(key_file_pointer == NULL) {
        printf("Key file path NULL \n");
    } else {
        printf("Key file path OK \n");
    }

    FILE* result_file_pointer;
    result_file_pointer = fopen("data/data_result.txt", "w");
    if(result_file_pointer == NULL) {
        printf("Result file path NULL \n");
    } else {
        printf("Result file path OK \n");
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
    
    printf("Starting insert jobs\n");
    for(int i = 0; i < SEARCH_TO_INSERT; i++) {
        // Insert keys
        fill_insert_buffer(&key_value_file_pointer, &cpu_buffer, &slab);
        //print_buffer(&cpu_buffer);
        transfer_data(&cpu_buffer, &gpu_buffer, HOST_TO_DEVICE);
        insert_gpu(&gpu_buffer, &hash_table);
    }
    printf("Finished insert jobs\n");

    printf("Starting search jobs\n");
    // Retrieve keys
    fill_search_buffer(&key_file_pointer, &cpu_buffer);
    //print_buffer(&cpu_buffer);
    transfer_data(&cpu_buffer, &gpu_buffer, HOST_TO_DEVICE);
    search_gpu(&gpu_buffer, &hash_table);  
    transfer_data(&cpu_buffer, &gpu_buffer, DEVICE_TO_HOST);
    //print_buffer(&cpu_buffer);
    write_search_data_to_file(&cpu_buffer, &result_file_pointer, &slab);
    printf("Finished search jobs\n");

    //******************************************************************************
    fclose(key_value_file_pointer);
    fclose(key_file_pointer);
    fclose(result_file_pointer);
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
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Reset failed");
        return 1;
    }
    return 0;
}

void insert_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table) { 
    int nr_insert_keys = MAX_INSERT_JOBS;
    int nr_threads = nr_insert_keys * BUCKET_SIZE;
    int BLOCKS = nr_threads / THREADS_PER_BLOCK;
    int THREADS = THREADS_PER_BLOCK;
    gpu_insert_key <<< BLOCKS, THREADS>>> ( (*gpu_buffer)->key_insert , (*hash_table), nr_insert_keys);
    cudaDeviceSynchronize();
}

void search_gpu(elem_buffer_t** gpu_buffer, bucket_t** hash_table) { 
    int nr_search_keys = MAX_SEARCH_JOBS;
    int nr_threads = nr_search_keys * BUCKET_SIZE;
    int BLOCKS = nr_threads / THREADS_PER_BLOCK;
    int THREADS = THREADS_PER_BLOCK;
    gpu_search_key <<< BLOCKS, THREADS>>> ( (*gpu_buffer)->key_search , (*gpu_buffer)->locations,  (*hash_table), nr_search_keys);
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
        // Compute hash and write to buffer
        uint64_t hash_result; 
        hash_result = *(uint64_t*) (&key_value_buffer[0]); // evil char to uint64 bit level hacking
        for(int j = 8; j <= KEY_LEN - 8; j+= 8) {
            hash_result = hash_result ^ *(uint64_t*) (&key_value_buffer[j]);
        }
        //printf("Hash result for insert key %sis 0x%I64x\n", key_value_buffer, hash_result);
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
    (*cpu_buffer)->nr_search_keys = 0;
    (*cpu_buffer)->nr_delete_keys = 0;
    (*slab)[slab_index].occupied = 1;
}

void fill_search_buffer(FILE** file_pointer, elem_buffer_t** cpu_buffer) { 
    char key_value_buffer[FILE_BUFFER_LEN];
    for(int i = 0; i < MAX_SEARCH_JOBS; i++) {
        //***************************************************************************
        // Compute hash and write to buffer
        fgets(key_value_buffer, FILE_BUFFER_LEN, *file_pointer);
        uint64_t hash_result; 
        hash_result = *(uint64_t*) (&key_value_buffer[0]); // evil char to uint64 bit level hacking
        for(int j = 8; j <= KEY_LEN - 8; j+= 8) {
            hash_result = hash_result ^ *(uint64_t*) (&key_value_buffer[j]);
        }
        //printf("Hash result for search key %sis 0x%I64x\n", key_value_buffer, hash_result);
        (*cpu_buffer)->key_search[i].hash      = (hash_t)       (hash_result >> 32);
        (*cpu_buffer)->key_search[i].signature = (signature_t)  (hash_result      );
        //***************************************************************************
    }
    (*cpu_buffer)->nr_search_keys = MAX_SEARCH_JOBS;
    (*cpu_buffer)->nr_insert_keys = 0;
    (*cpu_buffer)->nr_delete_keys = 0;
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

void write_search_data_to_file(elem_buffer_t** cpu_buffer, FILE** file_pointer,  slab_block_t* slab[]) {
    for(int i = 0; i < MAX_SEARCH_JOBS; i++) { 
        char key[KEY_LEN+1];
        char value[VALUE_LEN+1];
        int slab_item = (*cpu_buffer)->locations[i] & SLAB_ITEM_MASK;
        int slab_block = (*cpu_buffer)->locations[i] >> NR_SLAB_ITEMS_LOG;
        //printf("Accessing slab block %0d item %0d \n", slab_block, slab_item);
        strncpy(key, (*slab)[slab_block].item[slab_item].key, KEY_LEN);
        strncpy(value, (*slab)[slab_block].item[slab_item].value, VALUE_LEN);
        //printf("Key   %s\nValue %s\n", key, value);
        fwrite(key   , 1 , sizeof(key) -1   , *file_pointer);
        fwrite("\n"  , 1 , sizeof(char)  , *file_pointer);
        fwrite(value , 1 , sizeof(value) -1  , *file_pointer);
        fwrite("\n"  , 1 , sizeof(char)  , *file_pointer);
    }
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
        printf("0x%x\n",(*cpu_buffer)->locations[i] );
    }

}