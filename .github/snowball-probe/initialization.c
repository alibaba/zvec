#include <stdlib.h>
#include "header.h"

#ifdef _MSC_VER
__declspec(noinline)
#endif
static struct name *make_name(byte *s) {
    NEW(name, p);
    *p = (struct name){0};
    p->mode = m_unknown;
    p->s = copy_s(s);
    p->type = t_routine;
    p->count = -1;
    return p;
}

int main(void) {
    char filename[] = "initialization.sbl";
    byte *input = create_s(0);
    struct tokeniser *t = create_tokeniser(input, filename);
    struct name *n = make_name(input);
    int ok = n->mode == m_unknown && t->file == filename;
    printf("mode=%d (expected %d), file_preserved=%d, line=%d\n",
           n->mode, m_unknown, t->file == filename, t->line_number);
    lose_s(n->s);
    check_free(n);
    close_tokeniser(t);
    return ok ? 0 : 1;
}
