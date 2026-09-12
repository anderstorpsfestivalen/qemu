#!/usr/bin/env python3
"""Actual conservative TB coverage mask: no missed byte, bounded shifts."""
from pathlib import Path
import os
import subprocess
import tempfile
root = Path(__file__).resolve().parents[2]
source = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "tb-code-bitmap.h"
int main(void)
{
    const unsigned page_bits[] = {6,7,8,10,12,14,16,21,63};
    for(unsigned p=0;p<sizeof(page_bits)/sizeof(page_bits[0]);++p) {
        unsigned bits=page_bits[p];
        uint64_t unit=UINT64_C(1)<<(bits-6);
        /* Every possible pair of code buckets, with both edge bytes. */
        for(unsigned first=0;first<64;++first) for(unsigned last=first;last<64;++last) {
            uint64_t start=first*unit,end=last*unit+unit-1;
            uint64_t mask=tb_code_range_mask(bits,start,end);
            for(unsigned byte=0;byte<64;++byte) {
                uint64_t a=tb_code_range_mask(bits,byte*unit,byte*unit);
                uint64_t b=tb_code_range_mask(bits,byte*unit+unit-1,byte*unit+unit-1);
                assert(a==b && a==(UINT64_C(1)<<byte));
                assert(!!(mask&a)==(byte>=first && byte<=last));
            }
        }
        assert(tb_code_range_mask(bits,0,UINT64_MAX)==UINT64_MAX);
    }
    /* Real4KiB-page byte overlaps, including writes straddling a bucket. */
    for(unsigned start=0;start<4096;++start) for(unsigned length=1;length<=128 && start+length<=4096;++length) {
        uint64_t code=tb_code_range_mask(12,0x100000+start,0x100000+start+length-1);
        for(unsigned n=0;n<length;++n)
            assert(code&tb_code_range_mask(12,0x100000+start+n,0x100000+start+n));
    }
    assert(tb_code_range_mask(5,0,1)==UINT64_MAX);
    assert(tb_code_range_mask(64,0,1)==UINT64_MAX);
    assert(tb_code_range_mask(12,8,7)==UINT64_MAX);
    assert(tb_code_range_mask(12,4095,4096)==UINT64_MAX);
    puts("PASS actual TB coverage mask: all64bucket ranges,6..63bitpages,byte overlap,invalid/cross-page conservative fallback");
}
'''
with tempfile.TemporaryDirectory(prefix='tcg-code-bitmap-') as temporary:
    path=Path(temporary);(path/'test.c').write_text(source)
    subprocess.run([os.environ.get('CC','cc'),'-std=c99','-O2','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined','-I'+str(root/'accel/tcg'),
                    str(path/'test.c'),'-o',str(path/'test')],check=True)
    subprocess.run([str(path/'test')],check=True)
