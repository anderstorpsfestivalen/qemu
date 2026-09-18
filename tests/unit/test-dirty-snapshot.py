#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Execute the actual dirty snapshot body and atomic bitmap copy with seams."""
from pathlib import Path
import os
import subprocess
import tempfile
root = Path(__file__).resolve().parents[2]
def function(path, name):
    text = path.read_text()
    start = text.index(name)
    start = text.rfind('\n', 0, start) + 1
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]
copy = function(root/'util/bitmap.c', 'void bitmap_copy_and_clear_atomic')
empty = function(root/'util/bitmap.c', 'int slow_bitmap_empty')
clean = function(root/'system/physmem.c', 'bool physical_memory_snapshot_is_clean')
wrapper = function(root/'system/memory.c', 'DirtyBitmapSnapshot *memory_region_snapshot_and_clear_dirty')
body = function(root/'system/physmem.c', 'DirtyBitmapSnapshot *physical_memory_snapshot_and_clear_dirty')
source = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
typedef uint64_t ram_addr_t;
typedef uint64_t hwaddr;
#define TARGET_PAGE_BITS 12
#define BITS_PER_LONG (sizeof(unsigned long)*CHAR_BIT)
#define BITS_PER_LEVEL (sizeof(unsigned long)==8 ? 6 : 5)
#define DIRTY_MEMORY_BLOCK_SIZE (BITS_PER_LONG*2)
#define QEMU_ALIGN_DOWN(n,a) ((n)&~((uint64_t)(a)-1))
#define QEMU_ALIGN_UP(n,a) QEMU_ALIGN_DOWN((n)+(a)-1,a)
#define QEMU_IS_ALIGNED(n,a) (!((n)&((a)-1)))
#define BITMAP_LAST_WORD_MASK(n) (~0UL >> ((- (n)) & (BITS_PER_LONG - 1)))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define RAM_ADDR_INVALID UINT64_MAX
#define g_malloc0(n) calloc(1,n)
typedef struct {ram_addr_t base; void *ram_block;} MemoryRegion;
typedef struct {unsigned long *blocks[4];} DirtyMemoryBlocks;
typedef struct {ram_addr_t start,end;unsigned long dirty[];} DirtyBitmapSnapshot;
static struct {DirtyMemoryBlocks *dirty_memory[2];} ram_list;
static unsigned lock_depth, resets, callbacks, exchanges;
static ram_addr_t expected_start,expected_length;
static hwaddr expected_offset;
static unsigned long *inject_at;
static unsigned long inject_bits;
#define WITH_RCU_READ_LOCK_GUARD() for(bool guard=(++lock_depth,true);guard;guard=false,--lock_depth)
#define qatomic_rcu_read(p) (*(p))
static unsigned long exchange(unsigned long *p,unsigned long v) {
 assert(lock_depth==1);++exchanges;
 unsigned long old=__atomic_exchange_n(p,v,__ATOMIC_SEQ_CST);
 /* Deterministic writer immediately after the atomic snapshot exchange. */
 if(p==inject_at){__atomic_fetch_or(p,inject_bits,__ATOMIC_SEQ_CST);inject_at=NULL;}
 return old;
}
#define qatomic_xchg(p,v) exchange(p,v)
static ram_addr_t memory_region_get_ram_addr(MemoryRegion*m){return m->base;}
static void physical_memory_dirty_bits_cleared(ram_addr_t s,ram_addr_t n){assert(!lock_depth&&s==expected_start&&n==expected_length);++resets;}
static void memory_region_clear_dirty_bitmap(MemoryRegion*m,hwaddr o,hwaddr n){assert(!lock_depth&&m->base+o==expected_start&&o==expected_offset&&n==expected_length);++callbacks;}
'''+empty+r'''
#define bitmap_empty slow_bitmap_empty
'''+copy+'\n'+clean+'\n'+body+r'''
static bool tcg = true;
static unsigned fences, syncs;
static bool tcg_enabled(void){return tcg;}
static void memory_region_sync_dirty_bitmap(MemoryRegion*m,bool last){assert(m&&m->ram_block&&!last);++syncs;}
static void memory_global_after_dirty_log_sync(void){++fences;}
'''+wrapper+r'''
static unsigned long storage[4][2];
static DirtyMemoryBlocks blocks;
static MemoryRegion mr={.base=0,.ram_block=(void*)1};
static DirtyBitmapSnapshot *snapshot(hwaddr off,hwaddr n){
 expected_offset=off;expected_start=mr.base+off;expected_length=n;
 unsigned c=callbacks,f=fences,y=syncs;DirtyBitmapSnapshot*s=memory_region_snapshot_and_clear_dirty(&mr,off,n,0);assert(callbacks==c+1&&!lock_depth&&syncs==y+1);assert(fences==f+(!tcg||!physical_memory_snapshot_is_clean(s)));return s;
}
int main(void){
 for(unsigned i=0;i<4;i++){blocks.blocks[i]=storage[i];}
 ram_list.dirty_memory[0]=&blocks;
 const hwaddr word_bytes=BITS_PER_LONG<<TARGET_PAGE_BITS;
 DirtyBitmapSnapshot*s=snapshot(0,word_bytes);assert(!resets&&exchanges==1&&s->dirty[0]==0);free(s);
 /* Zero-length aligned range: no read from the empty flexible array. */
 unsigned e=exchanges;s=snapshot(0,0);assert(!resets&&exchanges==e&&s->start==s->end);free(s);
 /* First/last bit and aligned superset edges across dirty-memory blocks. */
 for(unsigned bit=0;bit<6*BITS_PER_LONG;bit++){
  unsigned block=bit/DIRTY_MEMORY_BLOCK_SIZE,word=(bit%DIRTY_MEMORY_BLOCK_SIZE)/BITS_PER_LONG;
  unsigned long mask=1UL<<(bit%BITS_PER_LONG);storage[block][word]=mask;
  unsigned r=resets;s=snapshot(1,6*word_bytes-2);assert(resets==r+1);
  assert(s->dirty[bit/BITS_PER_LONG]==mask&&storage[block][word]==0);free(s);
 }
 /* Rounded capture includes bits outside the requested page, conservatively. */
 storage[0][0]=1;s=snapshot(4096,4096);assert(s->dirty[0]==1);free(s);
 /* A new write after a clean atomic exchange remains dirty for next pass.
  * It neither changes the private snapshot nor needs an unnecessary reset. */
 inject_at=&storage[0][0];inject_bits=8;unsigned r=resets;
 s=snapshot(0,word_bytes);assert(!s->dirty[0]&&storage[0][0]==8&&resets==r);free(s);
 s=snapshot(0,word_bytes);assert(s->dirty[0]==8&&!storage[0][0]&&resets==r+1);free(s);
 /* Existing captured dirty bit plus a subsequent write: reset once, retain new. */
 storage[0][0]=2;inject_at=&storage[0][0];inject_bits=16;r=resets;
 s=snapshot(0,word_bytes);assert(s->dirty[0]==2&&storage[0][0]==16&&resets==r+1);free(s);
 s=snapshot(0,word_bytes);assert(s->dirty[0]==16&&!storage[0][0]);free(s);
 /* Nonzero MemoryRegion base and offset retain exact callback bounds. */
 mr.base=word_bytes;storage[0][1]=32;r=resets;s=snapshot(4096,4096);assert(resets==r+1&&s->start==word_bytes&&s->dirty[0]==32);free(s);
 /* Other accelerators still notify every listener for empty and dirty captures. */
 tcg=false;s=snapshot(0,word_bytes);free(s);
 storage[0][1]=64;s=snapshot(0,word_bytes);assert(s->dirty[0]==64);free(s);
 puts("PASS actual dirty snapshot and MemoryRegion fence: clean, dirty, all word/block edges, empty, rounded range, post-exchange writers, unconditional hardware callback");
}
'''
with tempfile.TemporaryDirectory(prefix='dirty-snapshot-') as temporary:
    p=Path(temporary);(p/'test.c').write_text(source)
    subprocess.run([os.environ.get('CC','cc'),'-std=gnu11','-O2','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(p/'test.c'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
