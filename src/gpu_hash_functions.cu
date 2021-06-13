#include <stdint.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gpu_hash_functions.h"

__global__ void gpu_search_key(
    key_search_t  *keys,
    location_t  *locations,
    bucket_t    *hash_map,
    int          no_items
    ) {

    int idx           = blockIdx.x * blockDim.x + threadIdx.x; 
    int id            = idx / BUCKET_SIZE;  // assign BUCKET_SIZE threads to a single element
    int bucket_offset = idx % BUCKET_SIZE;  
    
    key_search_t *key = &(keys[id]);
    
    int first_hash = (key->hash) & HASH_TABLE_MASK ; 
    bucket_t *bucket = &(hash_map[first_hash]);
    
    printf("SEARCH_FIRST_HASH: Thread %d block %d ID %d bucket id %d index %d bucket hash 0x%x signature 0x%x\n", threadIdx.x, blockIdx.x, idx, id, bucket_offset, first_hash, key->signature );

    if (bucket->signature[bucket_offset] == key->signature) {
        locations[id] = bucket->location[bucket_offset];
        if(DEBUG_PRINT)
            printf("SEARCH_FIRST_BUCKET: Thread %d block %d ID %d bucket id %d index %d bucket hash 0x%x signature 0x%x location 0x%x\n", threadIdx.x, blockIdx.x, idx, id, bucket_offset, first_hash, key->signature, bucket->location[bucket_offset] );
    }
    __syncthreads();

    // each thread has to get the state of its group of threads inside the warp
    int warp_id = idx % WARP_SIZE;
    int ballot_shift =  ((warp_id/BUCKET_SIZE)*BUCKET_SIZE);
    int ballot_mask  = (0xFF << ballot_shift);
    int warp_state = __ballot_sync(ballot_mask, bucket->signature[bucket_offset] == key->signature ) >> ballot_shift;

    if(DEBUG_PRINT)
        printf("SEARCH_CHECK_FOUND: Thread %d of block %d, my ID is %d. Inside my warp my ID is %d and I have been assigned to look in bucket %d, index %d. \nThe mask I use to look for my thread group is 0x%x and the vote returned 0x%x\n\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, ballot_mask, warp_state );

    if(warp_state == 0) {
        int second_hash = (key->hash ^ key->signature) & HASH_TABLE_MASK;
        bucket = &(hash_map[second_hash]);
        printf("SEARCH_SECOND_HASH: Thread %d block %d ID %d bucket id %d index %d bucket hash 0x%x signature 0x%x\n", threadIdx.x, blockIdx.x, idx, id, bucket_offset, second_hash, key->signature );
        if (bucket->signature[bucket_offset] == key->signature) {
            locations[id] = bucket->location[bucket_offset];
            if(DEBUG_PRINT)
                printf("SEARCH_SECOND_BUCKET: Thread %d block %d ID %d warp_id %d bucket id %d index %d bucket hash 0x%x signature 0x%x location 0x%x\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, second_hash, key->signature, bucket->location[bucket_offset] );
        }
    }

    return;
} 

__global__ void gpu_insert_key(
    key_insert_t  *keys,
    bucket_t    *hash_map,
    int          no_items
    ) {
    
    int idx           = blockIdx.x * blockDim.x + threadIdx.x; 
    int id            = idx / BUCKET_SIZE;  // assign BUCKET_SIZE threads to a single element
    int bucket_offset = idx % BUCKET_SIZE;  
    int chosen_thread_id;

    int first_hash, second_hash, temp_signature, temp_location;
    int cuckoo_counter;

    key_insert_t *key = &(keys[id]);
    first_hash = (key->hash) & HASH_TABLE_MASK;
    bucket_t *bucket = &(hash_map[first_hash]);
    
    // each thread has to get the state of its group of threads inside the warp
    int warp_id = idx % WARP_SIZE;
    int ballot_shift =  ((warp_id/BUCKET_SIZE)*BUCKET_SIZE);
    int ballot_mask  = (0xFF << ballot_shift);
    int warp_state = __ballot_sync(ballot_mask, bucket->signature[bucket_offset] == 0) >> ballot_shift;
    
    if(DEBUG_PRINT)
        printf("INSERT_FIRST_HASH: I am thread %d of block %d, my ID is %d. Inside my warp my ID is %d and I have been assigned to look in bucket %d, index %d. The mask I use to look for my thread group is 0x%x and the vote returned 0x%x.\n\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, ballot_mask, warp_state );

    if(warp_state != 0) {
        chosen_thread_id = __ffs(warp_state) - 1; 
    
        if(bucket_offset == chosen_thread_id) {
            bucket->signature[bucket_offset] = key->signature;
            bucket->location[bucket_offset] = key->location;
            if(DEBUG_PRINT)
                printf("INSERT_FIRST_BUCKET: Thread %d block %d ID %d warp_id %d bucket id %d index %d bucket hash 0x%x signature 0x%x location 0x%x\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, first_hash, key->signature, key->location );
        }
        __syncthreads();
        
        goto end;
    }

    cuckoo_counter = 0;

cuckoo_insert:
    second_hash = (key->hash ^ key->signature) & HASH_TABLE_MASK;
    bucket = &(hash_map[second_hash]);
    warp_state = __ballot_sync(ballot_mask, bucket->signature[bucket_offset] == 0) >> ballot_shift;
    
    if(DEBUG_PRINT)
        printf("INSERT_SECOND_HASH: I am thread %d of block %d, my ID is %d. Inside my warp my ID is %d and I have been assigned to look in bucket %d, index %d. The mask I use to look for my thread group is 0x%x and the vote returned 0x%x.\n\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, ballot_mask, warp_state );

    if(warp_state != 0) {
        chosen_thread_id = __ffs(warp_state) - 1; 
        if(bucket_offset == chosen_thread_id) {
            bucket->signature[bucket_offset] = key->signature;
            bucket->location[bucket_offset] = key->location;
            if(DEBUG_PRINT)    
                printf("INSERT_SECOND_BUCKET: Thread %d block %d ID %d warp_id %d bucket id %d index %d bucket hash 0x%x signature 0x%x location 0x%x\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, second_hash, key->signature, key->location );
        }
        __syncthreads();
        goto end;
    } 

    // choose which item to replace based on signature
    chosen_thread_id = key->signature & ((1<<BUCKET_SIZE_LOG)-1);
    temp_signature = bucket->signature[chosen_thread_id];
    temp_location = bucket->location[chosen_thread_id];

    __syncthreads();

    cuckoo_counter++;
    if(bucket_offset == chosen_thread_id) { 
        if(DEBUG_PRINT)
            printf("REPLACE_SECOND_BUCKET: Thread %d block %d ID %d warp_id %d bucket id %d index %d bucket hash 0x%x signature 0x%x location 0x%x\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, second_hash, key->signature, key->location );
        bucket->signature[bucket_offset] = key->signature;
        bucket->location[bucket_offset] = key->location;
    }

    if(cuckoo_counter > CUCKOO_MAX) {
        key->signature = temp_signature;
        key->location = temp_location;
        goto cuckoo_insert;
    } else {
        goto end;
    }
end: 
    return;    
   
}



__global__ void gpu_delete_key(
    key_search_t  *keys,
    bucket_t    *hash_map,
    int          no_items
    ) {

    int idx           = blockIdx.x * blockDim.x + threadIdx.x; 
    int id            = idx / BUCKET_SIZE;  // assign BUCKET_SIZE threads to a single element
    int bucket_offset = idx % BUCKET_SIZE;  
    
    key_search_t *key = &(keys[id]);
    
    int first_hash = (key->hash) & HASH_TABLE_MASK;
    bucket_t *bucket = &(hash_map[first_hash]);
    if (bucket->signature[bucket_offset] == key->signature) {
        bucket->location[bucket_offset] = 0;
        bucket->signature[bucket_offset] = 0;
        if(DEBUG_PRINT)
            printf("DELETE_FIRST_BUCKET: Thread %d block %d ID %d bucket id %d index %d bucket hash 0x%x signature 0x%x \n", threadIdx.x, blockIdx.x, idx, id, bucket_offset, first_hash, key->signature );
    }
    __syncthreads();

    // each thread has to get the state of its group of threads inside the warp
    int warp_id = idx % WARP_SIZE;
    int ballot_shift =  ((warp_id/BUCKET_SIZE)*BUCKET_SIZE);
    int ballot_mask  = (0xFF << ballot_shift);
    int warp_state = __ballot_sync(ballot_mask, bucket->signature[bucket_offset] == key->signature ) >> ballot_shift;

    if(DEBUG_PRINT)
        printf("DELETE: I am thread %d of block %d, my ID is %d. Inside my warp my ID is %d and I have been assigned to look in bucket %d, index %d. \nThe mask I use to look for my thread group is 0x%x and the vote returned 0x%x\n\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, ballot_mask, warp_state );

    if(warp_state == 0) {
        int second_hash = (key->hash ^ key->signature) & HASH_TABLE_MASK;
        bucket = &(hash_map[second_hash]);
        if (bucket->signature[bucket_offset] == key->signature) {
            bucket->location[bucket_offset] = 0;
            bucket->signature[bucket_offset] = 0;
            if(DEBUG_PRINT)
                printf("DELETE_SECOND_BUCKET: Thread %d block %d ID %d warp_id %d bucket id %d index %d bucket hash 0x%x signature 0x%x\n", threadIdx.x, blockIdx.x, idx, warp_id, id, bucket_offset, second_hash, key->signature );
        }
    }

    return;
} 


