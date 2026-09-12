#!/usr/bin/env python3
"""Compile actual texture dependency functions and verify context ordering."""
from pathlib import Path
import os
import subprocess
import tempfile
root = Path(__file__).resolve().parents[2]
source = (root / 'hw/display/juke-retro-gl-platform.c').read_text()
functions = source[source.index('static void texture_wait('):source.index('static JrgTexture *bound_texture(')]
shim = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
typedef uintptr_t GLsync;
#define GL_TIMEOUT_IGNORED UINT64_MAX
#define GL_SYNC_GPU_COMMANDS_COMPLETE 1
static unsigned waits, fences, deletes;
typedef struct { uint64_t serial; } JrgGLContext;
typedef struct { uint64_t version, writer_serial, waiter_serial; GLsync last_write; } JrgTexture;
static void glWaitSync(GLsync sync, unsigned flags, uint64_t timeout) {
    assert(sync && !flags && timeout == UINT64_MAX); ++waits;
}
static void glDeleteSync(GLsync sync) { assert(sync); ++deletes; }
static GLsync glFenceSync(unsigned condition, unsigned flags) {
    assert(condition == 1 && !flags); return ++fences;
}
'''
main = r'''
int main(void) {
    JrgGLContext a = {1}, b = {2}, c = {3}; JrgTexture texture = {0};
    texture_wait(&a,&texture); assert(!waits);
    texture_written(&a,&texture);
    for (unsigned i=0;i<1000;++i) texture_wait(&a,&texture);
    assert(!waits && fences==1 && !deletes);
    texture_wait(&b,&texture); texture_wait(&b,&texture); assert(waits==1);
    texture_wait(&a,&texture); assert(waits==1); /* Producer remains ordered. */
    texture_written(&a,&texture); texture_wait(&b,&texture); assert(waits==2);
    texture_wait(&c,&texture); assert(waits==3);
    texture_wait(&b,&texture); assert(waits==4); /* Conservative waiter eviction. */
    texture_written(&b,&texture); texture_wait(&b,&texture); assert(waits==4);
    texture_wait(&a,&texture); assert(waits==5);
    b.serial=4; texture_wait(&b,&texture); assert(waits==6); /* Address reused. */
    texture_written(&b,&texture); texture_wait(&b,&texture); assert(waits==6);
    assert(texture.version==4 && fences==4 && deletes==3);
    puts("PASS texture synchronization: same-context ordering, cross-context dependencies, new writes and identity reuse");
}
'''
with tempfile.TemporaryDirectory(prefix='jrg-texture-sync-') as temporary:
    directory = Path(temporary)
    (directory / 'test.c').write_text(shim + functions + main)
    subprocess.run([os.environ.get('CC','cc'),'-std=c99','-O2','-Wall','-Wextra','-Werror',
                    '-fsanitize=address,undefined',str(directory/'test.c'),'-o',str(directory/'test')],check=True)
    subprocess.run([str(directory/'test')],check=True)
