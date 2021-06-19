#include<stdint.h>

#define HASH_LEN           8        // 32b
#define SIGNATURE_LEN      8        // 32
#define BUCKET_LEN         8       // size of signature-location pair
#define BUCKET_SIZE        8        // 8 pairs of signature-location per bucket
#define BUCKET_SIZE_LOG    3        

#define HASH_TABLE_SIZE    (1<<30)    // 1GB
#define HASH_NO_BUCKETS    (HASH_TABLE_SIZE / (BUCKET_LEN * BUCKET_SIZE)) 
#define HASH_TABLE_MASK    (HASH_NO_BUCKETS) - 1 

#define WARP_SIZE          32

#define CUCKOO_MAX         3

#define MAX_SEARCH_JOBS    32768
#define MAX_INSERT_JOBS    512 
#define MAX_DELETE_JOBS    512

#define SEARCH_TO_INSERT   MAX_SEARCH_JOBS/MAX_INSERT_JOBS

#define THREADS_PER_BLOCK 1024

#define DEBUG_PRINT        0

typedef uint32_t hash_t;
typedef uint32_t signature_t; 
typedef uint32_t location_t; 

typedef struct key_search {
    hash_t       hash;
    signature_t  signature;
}key_search_t;

typedef struct key_delete {
    hash_t       hash;
    signature_t  signature;
}key_delete_t;

typedef struct key_insert {
    hash_t       hash;
    signature_t  signature;
    location_t   location;
}key_insert_t;

typedef struct bucket {
    signature_t  signature[BUCKET_SIZE];
    location_t   location[BUCKET_SIZE];
}bucket_t;
