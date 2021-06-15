#define SLAB_LEN           512
#define SLAB_LEN_LOG       8

#define NR_SLAB_ITEMS      512
#define NR_SLAB_ITEMS_LOG  9

#define SLAB_ITEM_MASK     NR_SLAB_ITEMS-1

#define KEY_LEN            64
#define VALUE_LEN          64
#define FILE_BUFFER_LEN    KEY_LEN + VALUE_LEN + 1 

typedef struct slab_item {
    char key[KEY_LEN+1];
    char value[VALUE_LEN+1];
} slab_item_t;

typedef struct slab_block { 
    slab_item_t item[NR_SLAB_ITEMS];
    int occupied;
} slab_block_t;