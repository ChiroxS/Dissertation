typedef struct elem_buffer {    
    
    key_search_t *key_search;
    key_insert_t *key_insert;
    key_delete_t *key_delete;

    location_t   *locations; 

    uint32_t nr_search_keys;
    uint32_t nr_insert_keys;
    uint32_t nr_delete_keys;

} elem_buffer_t;

typedef enum {
    HOST_TO_DEVICE, 
    DEVICE_TO_HOST
} direction_t;