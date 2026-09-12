#!/usr/bin/env python3
"""Actual bounded diagnostic aggregation, including saturated query vocabulary."""
from pathlib import Path
import os
import subprocess
import tempfile
root=Path(__file__).resolve().parents[2]
source=r'''
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "standard-headers/juke/retro-gpu.h"
#include "standard-headers/juke/retro-gl.h"
#define G_N_ELEMENTS(a) (sizeof(a)/sizeof((a)[0]))
static uint32_t ldl_le_p(const void *p) { const uint8_t *b=p;return b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24; }
#include "juke-retro-diagnostics.h"
static void word(uint8_t *r,unsigned offset,uint32_t value) {
    for (unsigned i=0;i<4;++i) r[offset+i]=(uint8_t)(value>>(i*8));
}
int main(void) {
    JrgDiagnostics d={0}; uint8_t r[96]={0};
    word(r,0,JRG_GL_DESKTOP);word(r,32,JRG_DESKTOP_READBACK);word(r,48,3);word(r,52,5);
    jrg_diagnostic_record(&d,r);
    assert(d.records==1 && d.desktop[JRG_DESKTOP_READBACK]==1 && d.desktop_bytes[JRG_DESKTOP_READBACK]==60);
    word(r,0,JRG_GL_QUERY);word(r,32,809);
    for (unsigned i=0;i<JRG_DIAGNOSTIC_QUERIES+1;++i) { word(r,36,i);jrg_diagnostic_record(&d,r); }
    assert(d.query_count==JRG_DIAGNOSTIC_QUERIES && d.query_overflow==1);
    word(r,36,0);jrg_diagnostic_record(&d,r);assert(d.queries[0].count==2 && d.query_overflow==1);
    word(r,0,JRG_GL_CALL);word(r,32,JRG_DIAGNOSTIC_FUNCTIONS);jrg_diagnostic_record(&d,r);
    assert(d.function_overflow==1 && d.functions[809]==JRG_DIAGNOSTIC_QUERIES+2);
    puts("PASS bounded diagnostics: desktop byte accounting, histogram overflow, retained query updates and function bounds");
}
'''
with tempfile.TemporaryDirectory(prefix='jrg-diagnostics-') as temporary:
    directory=Path(temporary);(directory/'test.c').write_text(source)
    subprocess.run([os.environ.get('CC','cc'),'-std=c99','-O2','-Wall','-Wextra','-Werror',
        '-fsanitize=address,undefined','-I'+str(root/'include'),'-I'+str(root/'hw/display'),
        str(directory/'test.c'),'-o',str(directory/'test')],check=True)
    subprocess.run([str(directory/'test')],check=True)
