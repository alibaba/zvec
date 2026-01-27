// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zvec/ailego/encoding/json/mod_json.h>

#ifndef MOD_JSON_TOKEN_DEFOPTS
#define MOD_JSON_TOKEN_DEFOPTS 0 /* default options of token */
#endif
#ifndef MOD_JSON_TOKEN_DEFOBJDEP
#define MOD_JSON_TOKEN_DEFOBJDEP 64 /* default objects depth of token */
#endif
#ifndef MOD_JSON_TOKEN_DEFARRDEP
#define MOD_JSON_TOKEN_DEFARRDEP 64 /* default arrays depth of token */
#endif
#ifndef MOD_JSON_STRING_DEFSIZE
#define MOD_JSON_STRING_DEFSIZE 32 /* default started size of string */
#endif
#ifndef MOD_JSON_ARRAY_DEFSIZE
#define MOD_JSON_ARRAY_DEFSIZE 32 /* default started size of array */
#endif
#ifndef MOD_JSON_OBJECT_DEFSIZE
#define MOD_JSON_OBJECT_DEFSIZE 32 /* default started size of object */
#endif

#ifndef mod_json_malloc
#define mod_json_malloc malloc
#endif
#ifndef mod_json_free
#define mod_json_free free
#endif

#ifdef __GNUC__
#define mod_json_likely(x) __builtin_expect(!!(x), 1)
#define mod_json_unlikely(x) __builtin_expect(!!(x), 0)
#else
#define mod_json_likely(x) (x)
#define mod_json_unlikely(x) (x)
#endif

#define mod_json_minus_if_ne_zero(COND) \
  if (mod_json_unlikely((COND) != 0)) return (-1)

#define mod_json_minus_if_false(COND) \
  if (mod_json_unlikely(!(COND))) return (-1)

#define mod_json_null_if_ne_zero(COND) \
  if (mod_json_unlikely((COND) != 0)) return (NULL)

#define mod_json_null_if_false(COND) \
  if (mod_json_unlikely(!(COND))) return (NULL)

#if defined(_MSC_VER)
#pragma warning(disable : 4200)
#define strtoull _strtoui64
#define snprintf(buf, size, format, ...) \
  _snprintf_s(buf, size, _TRUNCATE, format, ##__VA_ARGS__)
#endif
#define mod_json_utils_snprintf snprintf
#define mod_json_utils_strtoi strtoull
#define mod_json_utils_strtof strtod
#define mod_json_utils_strlen strlen

/*! JSON Token
 */
struct mod_json_token {
  mod_json_state_t state;
  mod_json_error_t error;
  mod_json_cchar_t *context;
  mod_json_size_t options;
  mod_json_size_t object_max_depth;
  mod_json_size_t array_max_depth;
  mod_json_size_t object_depth;
  mod_json_size_t array_depth;
  mod_json_event_t event_code;
  mod_json_event_proc event_proc;
  mod_json_void_t *param;
  mod_json_char_t tags[0];
};

typedef struct mod_json_parser mod_json_parser_t;

/*! JSON Parser
 */
struct mod_json_parser {
  mod_json_string_t *key;
  mod_json_value_t *val_null;
  mod_json_value_t *val_true;
  mod_json_value_t *val_false;
  mod_json_value_t *val_zero;
  mod_json_value_t *val_zerof;
  mod_json_value_t *val_empty;
  mod_json_value_t *vals[0];
};

static inline mod_json_size_t mod_json_utils_clp2(mod_json_size_t n) {
  n = n - 1;
  n = n | (n >> 1);
  n = n | (n >> 2);
  n = n | (n >> 4);
  n = n | (n >> 8);
  n = n | (n >> 16);
  return (n + 1);
}

static inline mod_json_size_t mod_json_utils_itostr(mod_json_char_t *buf,
                                                    mod_json_integer_t val) {
  mod_json_char_t *pos, *first, *last;

  pos = buf;
  if (val < 0) {
    *pos++ = '-';
    val = -val;
  }

  /* save pointer to first digit */
  first = pos;

  do {
    /* convert to ASCII and store */
    *pos++ = (mod_json_char_t)(val % 10 + '0');

    /* next digit */
    val /= 10;

  } while (val > 0);

  *pos = '\0';

  /* save pointer to last digit */
  last = pos - 1;

  /* reverse digit string */
  while (first < last) {
    mod_json_char_t temp = *first;
    *first++ = *last;
    *last-- = temp;
  }
  return (mod_json_size_t)(pos - buf);
}

static inline mod_json_float_t mod_json_utils_pow10(int n) {
  /* 1e-308...1e308: 617 * 8 bytes = 4936 bytes */
  static const mod_json_float_t etab[] = {
      1e-308, 1e-307, 1e-306, 1e-305, 1e-304, 1e-303, 1e-302, 1e-301, 1e-300,
      1e-299, 1e-298, 1e-297, 1e-296, 1e-295, 1e-294, 1e-293, 1e-292, 1e-291,
      1e-290, 1e-289, 1e-288, 1e-287, 1e-286, 1e-285, 1e-284, 1e-283, 1e-282,
      1e-281, 1e-280, 1e-279, 1e-278, 1e-277, 1e-276, 1e-275, 1e-274, 1e-273,
      1e-272, 1e-271, 1e-270, 1e-269, 1e-268, 1e-267, 1e-266, 1e-265, 1e-264,
      1e-263, 1e-262, 1e-261, 1e-260, 1e-259, 1e-258, 1e-257, 1e-256, 1e-255,
      1e-254, 1e-253, 1e-252, 1e-251, 1e-250, 1e-249, 1e-248, 1e-247, 1e-246,
      1e-245, 1e-244, 1e-243, 1e-242, 1e-241, 1e-240, 1e-239, 1e-238, 1e-237,
      1e-236, 1e-235, 1e-234, 1e-233, 1e-232, 1e-231, 1e-230, 1e-229, 1e-228,
      1e-227, 1e-226, 1e-225, 1e-224, 1e-223, 1e-222, 1e-221, 1e-220, 1e-219,
      1e-218, 1e-217, 1e-216, 1e-215, 1e-214, 1e-213, 1e-212, 1e-211, 1e-210,
      1e-209, 1e-208, 1e-207, 1e-206, 1e-205, 1e-204, 1e-203, 1e-202, 1e-201,
      1e-200, 1e-199, 1e-198, 1e-197, 1e-196, 1e-195, 1e-194, 1e-193, 1e-192,
      1e-191, 1e-190, 1e-189, 1e-188, 1e-187, 1e-186, 1e-185, 1e-184, 1e-183,
      1e-182, 1e-181, 1e-180, 1e-179, 1e-178, 1e-177, 1e-176, 1e-175, 1e-174,
      1e-173, 1e-172, 1e-171, 1e-170, 1e-169, 1e-168, 1e-167, 1e-166, 1e-165,
      1e-164, 1e-163, 1e-162, 1e-161, 1e-160, 1e-159, 1e-158, 1e-157, 1e-156,
      1e-155, 1e-154, 1e-153, 1e-152, 1e-151, 1e-150, 1e-149, 1e-148, 1e-147,
      1e-146, 1e-145, 1e-144, 1e-143, 1e-142, 1e-141, 1e-140, 1e-139, 1e-138,
      1e-137, 1e-136, 1e-135, 1e-134, 1e-133, 1e-132, 1e-131, 1e-130, 1e-129,
      1e-128, 1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120,
      1e-119, 1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111,
      1e-110, 1e-109, 1e-108, 1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102,
      1e-101, 1e-100, 1e-99,  1e-98,  1e-97,  1e-96,  1e-95,  1e-94,  1e-93,
      1e-92,  1e-91,  1e-90,  1e-89,  1e-88,  1e-87,  1e-86,  1e-85,  1e-84,
      1e-83,  1e-82,  1e-81,  1e-80,  1e-79,  1e-78,  1e-77,  1e-76,  1e-75,
      1e-74,  1e-73,  1e-72,  1e-71,  1e-70,  1e-69,  1e-68,  1e-67,  1e-66,
      1e-65,  1e-64,  1e-63,  1e-62,  1e-61,  1e-60,  1e-59,  1e-58,  1e-57,
      1e-56,  1e-55,  1e-54,  1e-53,  1e-52,  1e-51,  1e-50,  1e-49,  1e-48,
      1e-47,  1e-46,  1e-45,  1e-44,  1e-43,  1e-42,  1e-41,  1e-40,  1e-39,
      1e-38,  1e-37,  1e-36,  1e-35,  1e-34,  1e-33,  1e-32,  1e-31,  1e-30,
      1e-29,  1e-28,  1e-27,  1e-26,  1e-25,  1e-24,  1e-23,  1e-22,  1e-21,
      1e-20,  1e-19,  1e-18,  1e-17,  1e-16,  1e-15,  1e-14,  1e-13,  1e-12,
      1e-11,  1e-10,  1e-9,   1e-8,   1e-7,   1e-6,   1e-5,   1e-4,   1e-3,
      1e-2,   1e-1,   1e+0,   1e+1,   1e+2,   1e+3,   1e+4,   1e+5,   1e+6,
      1e+7,   1e+8,   1e+9,   1e+10,  1e+11,  1e+12,  1e+13,  1e+14,  1e+15,
      1e+16,  1e+17,  1e+18,  1e+19,  1e+20,  1e+21,  1e+22,  1e+23,  1e+24,
      1e+25,  1e+26,  1e+27,  1e+28,  1e+29,  1e+30,  1e+31,  1e+32,  1e+33,
      1e+34,  1e+35,  1e+36,  1e+37,  1e+38,  1e+39,  1e+40,  1e+41,  1e+42,
      1e+43,  1e+44,  1e+45,  1e+46,  1e+47,  1e+48,  1e+49,  1e+50,  1e+51,
      1e+52,  1e+53,  1e+54,  1e+55,  1e+56,  1e+57,  1e+58,  1e+59,  1e+60,
      1e+61,  1e+62,  1e+63,  1e+64,  1e+65,  1e+66,  1e+67,  1e+68,  1e+69,
      1e+70,  1e+71,  1e+72,  1e+73,  1e+74,  1e+75,  1e+76,  1e+77,  1e+78,
      1e+79,  1e+80,  1e+81,  1e+82,  1e+83,  1e+84,  1e+85,  1e+86,  1e+87,
      1e+88,  1e+89,  1e+90,  1e+91,  1e+92,  1e+93,  1e+94,  1e+95,  1e+96,
      1e+97,  1e+98,  1e+99,  1e+100, 1e+101, 1e+102, 1e+103, 1e+104, 1e+105,
      1e+106, 1e+107, 1e+108, 1e+109, 1e+110, 1e+111, 1e+112, 1e+113, 1e+114,
      1e+115, 1e+116, 1e+117, 1e+118, 1e+119, 1e+120, 1e+121, 1e+122, 1e+123,
      1e+124, 1e+125, 1e+126, 1e+127, 1e+128, 1e+129, 1e+130, 1e+131, 1e+132,
      1e+133, 1e+134, 1e+135, 1e+136, 1e+137, 1e+138, 1e+139, 1e+140, 1e+141,
      1e+142, 1e+143, 1e+144, 1e+145, 1e+146, 1e+147, 1e+148, 1e+149, 1e+150,
      1e+151, 1e+152, 1e+153, 1e+154, 1e+155, 1e+156, 1e+157, 1e+158, 1e+159,
      1e+160, 1e+161, 1e+162, 1e+163, 1e+164, 1e+165, 1e+166, 1e+167, 1e+168,
      1e+169, 1e+170, 1e+171, 1e+172, 1e+173, 1e+174, 1e+175, 1e+176, 1e+177,
      1e+178, 1e+179, 1e+180, 1e+181, 1e+182, 1e+183, 1e+184, 1e+185, 1e+186,
      1e+187, 1e+188, 1e+189, 1e+190, 1e+191, 1e+192, 1e+193, 1e+194, 1e+195,
      1e+196, 1e+197, 1e+198, 1e+199, 1e+200, 1e+201, 1e+202, 1e+203, 1e+204,
      1e+205, 1e+206, 1e+207, 1e+208, 1e+209, 1e+210, 1e+211, 1e+212, 1e+213,
      1e+214, 1e+215, 1e+216, 1e+217, 1e+218, 1e+219, 1e+220, 1e+221, 1e+222,
      1e+223, 1e+224, 1e+225, 1e+226, 1e+227, 1e+228, 1e+229, 1e+230, 1e+231,
      1e+232, 1e+233, 1e+234, 1e+235, 1e+236, 1e+237, 1e+238, 1e+239, 1e+240,
      1e+241, 1e+242, 1e+243, 1e+244, 1e+245, 1e+246, 1e+247, 1e+248, 1e+249,
      1e+250, 1e+251, 1e+252, 1e+253, 1e+254, 1e+255, 1e+256, 1e+257, 1e+258,
      1e+259, 1e+260, 1e+261, 1e+262, 1e+263, 1e+264, 1e+265, 1e+266, 1e+267,
      1e+268, 1e+269, 1e+270, 1e+271, 1e+272, 1e+273, 1e+274, 1e+275, 1e+276,
      1e+277, 1e+278, 1e+279, 1e+280, 1e+281, 1e+282, 1e+283, 1e+284, 1e+285,
      1e+286, 1e+287, 1e+288, 1e+289, 1e+290, 1e+291, 1e+292, 1e+293, 1e+294,
      1e+295, 1e+296, 1e+297, 1e+298, 1e+299, 1e+300, 1e+301, 1e+302, 1e+303,
      1e+304, 1e+305, 1e+306, 1e+307, 1e+308};
  return (n < -308 ? 0.0 : etab[n + 308]);
}

static inline mod_json_cchar_t *mod_json_utils_strskpb(mod_json_cchar_t *cstr) {
  static const mod_json_char_t blanks[256] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, '\t', '\n', '\v', '\f', '\r', 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    ' ',  0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    0,    0,    0,    0,    0, 0};

  while (*(blanks + *cstr)) {
    ++cstr;
  }
  return cstr;
}

static inline mod_json_cchar_t *mod_json_utils_strskpc1(
    mod_json_cchar_t *cstr) {
  mod_json_char_t c;

  while ((c = *cstr++) != '\0') {
    if (c == '\r' || c == '\n') {
      return mod_json_utils_strskpb(cstr);
    }
  }
  return (cstr - 1);
}

static inline mod_json_cchar_t *mod_json_utils_strskpc2(
    mod_json_cchar_t *cstr) {
  mod_json_char_t c;

  while ((c = *cstr++) != '\0') {
    /* asterisk, slash */
    if (c == '*' && *cstr == '/') {
      return mod_json_utils_strskpb(cstr + 1);
    }
  }
  return (cstr - 1);
}

static inline mod_json_cchar_t *mod_json_utils_strskp(mod_json_cchar_t *cstr) {
  cstr = mod_json_utils_strskpb(cstr);

  /* treat it as comments? */
  while (*cstr == '/') {
    mod_json_char_t c = *(cstr + 1); /* second char */

    if (c == '/') {
      /* two slashes */
      cstr = mod_json_utils_strskpc1(cstr + 2);
    } else if (c == '*') {
      /* slash, asterisk */
      cstr = mod_json_utils_strskpc2(cstr + 2);
    } else {
      /* invalid format */
      break;
    }
  }
  return cstr;
}

static inline int mod_json_utils_char2hex(mod_json_char_t ch) {
  static const mod_json_char_t char2hex[256] = {
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  16, 16, 16, 16, 16, 16, 16, 10, 11, 12, 13, 14, 15, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16};
  return *(char2hex + ch);
}

static inline mod_json_cchar_t *mod_json_utils_strfquo(mod_json_cchar_t *cstr,
                                                       mod_json_char_t quo) {
  mod_json_char_t c;

  for (c = *cstr; c != quo; c = *(++cstr)) {
    if ((mod_json_uchar_t)c <= 0x1f) {
      return NULL;
    }

    if (c != '\\') {
      continue;
    }

    /* next char */
    switch (*(++cstr)) {
      case '\"':
      case '/':
      case 'b':
      case 'f':
      case '\\':
      case 'n':
      case 'r':
      case 't':
        /* ignore next char */
        break;

      case 'u':
        if (mod_json_utils_char2hex(*(cstr + 1)) > 15) {
          return NULL;
        }
        if (mod_json_utils_char2hex(*(cstr + 2)) > 15) {
          return NULL;
        }
        if (mod_json_utils_char2hex(*(cstr + 3)) > 15) {
          return NULL;
        }
        if (mod_json_utils_char2hex(*(cstr + 4)) > 15) {
          return NULL;
        }
        cstr += 4;
        break;

      default:
        /* invalid */
        return NULL;
    }
  }
  /* found it */
  return cstr;
}

static inline mod_json_cchar_t *mod_json_utils_strfquo2(mod_json_cchar_t *cstr,
                                                        mod_json_char_t quo) {
  mod_json_char_t c;

  for (c = *cstr; c; c = *(++cstr)) {
    if (c == quo) {
      /* found it */
      return cstr;
    }

    if (c == '\\') {
      /* ignore next char */
      if (*(++cstr) == '\0') {
        break;
      }
    }
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_utils_strfsep(mod_json_cchar_t *cstr) {
  mod_json_char_t c;

  while ((c = *cstr++) != '\0') {
    switch (c) {
      case ':':
      case ' ':
      case '\t':
      case '\r':
      case '\n':
      case '\f':
      case '\v':
        return (cstr - 1);
    }
  }
  return (cstr - 1);
}

static inline mod_json_cchar_t *mod_json_utils_strfsep2(
    mod_json_cchar_t *cstr) {
  mod_json_char_t c;

  while ((c = *cstr++) != '\0') {
    switch (c) {
      case ':':
      case ' ':
      case '\t':
      case '\r':
      case '\n':
      case '\f':
      case '\v':
        return (cstr - 1);

      case '/':
        if (*cstr == '/' || *cstr == '*') {
          return (cstr - 1);
        }
    }
  }
  return (cstr - 1);
}

static inline mod_json_char_t *mod_json_utils_uni2utf8(mod_json_char_t *buf,
                                                       mod_json_size_t size,
                                                       mod_json_uchar_t high,
                                                       mod_json_uchar_t low) {
  /* convert to UTF-8 */
  if (high >= 0x8) {
    /* 0800 - FFFF | 1110xxxx 10xxxxxx 10xxxxxx */
    if (size >= 3) {
      *buf++ = (mod_json_char_t)(0xE0 | (high >> 4));
      *buf++ = (mod_json_char_t)(0x80 | ((high & 0xF) << 2) | (low >> 6));
      *buf++ = (mod_json_char_t)(0x80 | (low & 0x3F));
      return buf;
    }
  } else if (high > 0 || low >= 0x80) {
    /* 0080 - 07FF | 110xxxxx 10xxxxxx */
    if (size >= 2) {
      *buf++ = (mod_json_char_t)(0xC0 | (high << 2) | (low >> 6));
      *buf++ = (mod_json_char_t)(0x80 | (low & 0x3F));
      return buf;
    }
  } else {
    /* 0000 - 007F | 0xxxxxxx */
    if (size >= 1) {
      *buf++ = (mod_json_char_t)(low);
      return buf;
    }
  }
  return (mod_json_char_t *)0;
}

mod_json_value_t *mod_json_value_set_null(void) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_null;
  val->data.c_int = 0;
  return val;
}

mod_json_value_t *mod_json_value_set_object(mod_json_object_t *obj) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_object;
  val->data.c_obj = obj ? mod_json_object_grab(obj) : NULL;
  return val;
}

mod_json_value_t *mod_json_value_set_array(mod_json_array_t *arr) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_array;
  val->data.c_arr = arr ? mod_json_array_grab(arr) : NULL;
  return val;
}

mod_json_value_t *mod_json_value_set_string(mod_json_string_t *str) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_string;
  val->data.c_str = str ? mod_json_string_grab(str) : NULL;
  return val;
}

mod_json_value_t *mod_json_value_set_buffer(mod_json_cchar_t *buf,
                                            mod_json_size_t len) {
  mod_json_value_t *val;
  mod_json_string_t *str;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  /* create a string */
  str = mod_json_string_set(buf, len);
  if (mod_json_unlikely(!str)) {
    mod_json_free(val);
    return NULL;
  }

  val->refer = 1;
  val->type = mod_json_type_string;
  val->data.c_str = str;
  return val;
}

mod_json_value_t *mod_json_value_set_integer(mod_json_integer_t num) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_integer;
  val->data.c_int = num;
  return val;
}

mod_json_value_t *mod_json_value_set_float(mod_json_float_t dbl) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_float;
  val->data.c_float = dbl;
  return val;
}

mod_json_value_t *mod_json_value_set_boolean(mod_json_boolean_t bol) {
  mod_json_value_t *val;

  /* create a value */
  val = (mod_json_value_t *)mod_json_malloc(sizeof(mod_json_value_t));
  mod_json_null_if_false(val);

  val->refer = 1;
  val->type = mod_json_type_boolean;
  val->data.c_bool = bol ? MOD_JSON_TRUE : MOD_JSON_FALSE;
  return val;
}

static inline void mod_json_value_clear(mod_json_value_t *val) {
  switch (val->type) {
    case mod_json_type_object:
      mod_json_object_unset(val->data.c_obj);
      break;

    case mod_json_type_array:
      mod_json_array_unset(val->data.c_arr);
      break;

    case mod_json_type_string:
      mod_json_string_unset(val->data.c_str);
      break;

    default:
      break;
  }
}

void mod_json_value_assign_null(mod_json_value_t *val) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_null;
    val->data.c_int = 0;
  }
}

void mod_json_value_assign_object(mod_json_value_t *val,
                                  mod_json_object_t *obj) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_object;
    val->data.c_obj = obj ? mod_json_object_grab(obj) : NULL;
  }
}

void mod_json_value_assign_array(mod_json_value_t *val, mod_json_array_t *arr) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_array;
    val->data.c_arr = arr ? mod_json_array_grab(arr) : NULL;
  }
}

void mod_json_value_assign_string(mod_json_value_t *val,
                                  mod_json_string_t *str) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_string;
    val->data.c_str = str ? mod_json_string_grab(str) : NULL;
  }
}

void mod_json_value_assign_integer(mod_json_value_t *val,
                                   mod_json_integer_t num) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_integer;
    val->data.c_int = num;
  }
}

void mod_json_value_assign_float(mod_json_value_t *val, mod_json_float_t dbl) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_float;
    val->data.c_float = dbl;
  }
}

void mod_json_value_assign_boolean(mod_json_value_t *val,
                                   mod_json_boolean_t bol) {
  if (val) {
    mod_json_value_clear(val);
    val->type = mod_json_type_boolean;
    val->data.c_bool = bol ? MOD_JSON_TRUE : MOD_JSON_FALSE;
  }
}

void mod_json_value_assign(mod_json_value_t *dst, mod_json_value_t *src) {
  if (!dst || dst == src) {
    return;
  }

  if (!src) {
    /* treat as JSON null */
    mod_json_value_assign_null(dst);
    return;
  }

  switch (src->type) {
    case mod_json_type_boolean:
      mod_json_value_assign_boolean(dst, src->data.c_bool);
      break;

    case mod_json_type_integer:
      mod_json_value_assign_integer(dst, src->data.c_int);
      break;

    case mod_json_type_float:
      mod_json_value_assign_float(dst, src->data.c_float);
      break;

    case mod_json_type_string:
      mod_json_value_assign_string(dst, src->data.c_str);
      break;

    case mod_json_type_array:
      mod_json_value_assign_array(dst, src->data.c_arr);
      break;

    case mod_json_type_object:
      mod_json_value_assign_object(dst, src->data.c_obj);
      break;

    default:
      mod_json_value_assign_null(dst);
      break;
  }
}

static inline int mod_json_value_merge_array(mod_json_value_t *val,
                                             mod_json_array_t *arr) {
  if (val->type != mod_json_type_array || !val->data.c_arr) {
    mod_json_value_assign_array(val, arr);
    return 0;
  }

  if (arr) {
    if (mod_json_array_is_shared(val->data.c_arr)) {
      mod_json_array_put(val->data.c_arr);
      val->data.c_arr = mod_json_array_clone(val->data.c_arr);
    }
    return mod_json_array_merge(val->data.c_arr, arr);
  }
  return 0;
}

static inline int mod_json_value_merge_object(mod_json_value_t *val,
                                              mod_json_object_t *obj) {
  if (val->type != mod_json_type_object || !val->data.c_obj) {
    mod_json_value_assign_object(val, obj);
    return 0;
  }

  if (obj) {
    if (mod_json_object_is_shared(val->data.c_obj)) {
      mod_json_object_put(val->data.c_obj);
      val->data.c_obj = mod_json_object_clone(val->data.c_obj);
    }
    return mod_json_object_merge(val->data.c_obj, obj);
  }
  return 0;
}

int mod_json_value_merge(mod_json_value_t *dst, mod_json_value_t *src) {
  mod_json_minus_if_false(dst && dst != src);

  if (!src) {
    mod_json_value_assign_null(dst);
    return 0;
  }

  switch (src->type) {
    case mod_json_type_boolean:
      mod_json_value_assign_boolean(dst, src->data.c_bool);
      break;

    case mod_json_type_integer:
      mod_json_value_assign_integer(dst, src->data.c_int);
      break;

    case mod_json_type_float:
      mod_json_value_assign_float(dst, src->data.c_float);
      break;

    case mod_json_type_string:
      mod_json_value_assign_string(dst, src->data.c_str);
      break;

    case mod_json_type_array:
      return mod_json_value_merge_array(dst, src->data.c_arr);

    case mod_json_type_object:
      return mod_json_value_merge_object(dst, src->data.c_obj);

    default:
      mod_json_value_assign_null(dst);
      break;
  }
  return 0;
}

mod_json_object_t *mod_json_value_object(mod_json_value_t *val) {
  if (val && val->type == mod_json_type_object) {
    return (val->data.c_obj);
  }
  return NULL;
}

mod_json_array_t *mod_json_value_array(mod_json_value_t *val) {
  if (val && val->type == mod_json_type_array) {
    return (val->data.c_arr);
  }
  return NULL;
}

mod_json_string_t *mod_json_value_string(mod_json_value_t *val) {
  if (val && val->type == mod_json_type_string) {
    return (val->data.c_str);
  }
  return NULL;
}

mod_json_cchar_t *mod_json_value_cstring(mod_json_value_t *val) {
  if (val && val->type == mod_json_type_string) {
    return mod_json_string_cstr(val->data.c_str);
  }
  return NULL;
}

mod_json_float_t mod_json_value_float(mod_json_value_t *val) {
  if (val) {
    switch (val->type) {
      case mod_json_type_boolean:
        return (val->data.c_bool ? 1.0 : 0.0);

      case mod_json_type_integer:
        return (mod_json_float_t)(val->data.c_int);

      case mod_json_type_float:
        return (val->data.c_float);

      case mod_json_type_string:
        return mod_json_string_float(val->data.c_str);

      default:
        break;
    }
  }
  return (0.0);
}

mod_json_boolean_t mod_json_value_boolean(mod_json_value_t *val) {
  if (val) {
    switch (val->type) {
      case mod_json_type_null:
        return MOD_JSON_FALSE;

      case mod_json_type_object:
        return (mod_json_object_count(val->data.c_obj) != 0);

      case mod_json_type_array:
        return (mod_json_array_count(val->data.c_arr) != 0);

      case mod_json_type_string:
        return (mod_json_string_length(val->data.c_str) != 0);

      case mod_json_type_integer:
        return (val->data.c_int != 0);

      case mod_json_type_float:
        return (val->data.c_float != 0);

      case mod_json_type_boolean:
        return (val->data.c_bool);

      default:
        break;
    }
  }
  return MOD_JSON_FALSE;
}

mod_json_integer_t mod_json_value_integer(mod_json_value_t *val) {
  if (val) {
    switch (val->type) {
      case mod_json_type_boolean:
        return (val->data.c_bool ? 1 : 0);

      case mod_json_type_integer:
        return (val->data.c_int);

      case mod_json_type_float:
        return (mod_json_integer_t)(val->data.c_float);

      case mod_json_type_string:
        return mod_json_string_integer(val->data.c_str);

      default:
        break;
    }
  }
  return (0);
}

mod_json_value_t *mod_json_value_clone(mod_json_value_t *val) {
  if (val) {
    switch (val->type) {
      case mod_json_type_null:
        return mod_json_value_set_null();

      case mod_json_type_object:
        return mod_json_value_set_object(val->data.c_obj);

      case mod_json_type_array:
        return mod_json_value_set_array(val->data.c_arr);

      case mod_json_type_string:
        return mod_json_value_set_string(val->data.c_str);

      case mod_json_type_integer:
        return mod_json_value_set_integer(val->data.c_int);

      case mod_json_type_float:
        return mod_json_value_set_float(val->data.c_float);

      case mod_json_type_boolean:
        return mod_json_value_set_boolean(val->data.c_bool);

      default:
        break;
    }
  }
  return NULL;
}

static inline mod_json_boolean_t mod_json_value_is_equal_float(
    mod_json_float_t lhs, mod_json_float_t rhs) {
  mod_json_float_t diff = lhs - rhs;
  return ((diff < DBL_EPSILON) && (diff > -DBL_EPSILON));
}

mod_json_boolean_t mod_json_value_is_equal(mod_json_value_t *lhs,
                                           mod_json_value_t *rhs) {
  if (lhs == rhs) {
    /* The same pointer */
    return MOD_JSON_TRUE;
  }

  if (lhs && rhs && lhs->type == rhs->type) {
    switch (lhs->type) {
      case mod_json_type_null:
        return MOD_JSON_TRUE;

      case mod_json_type_object:
        return mod_json_object_is_equal(lhs->data.c_obj, rhs->data.c_obj);

      case mod_json_type_array:
        return mod_json_array_is_equal(lhs->data.c_arr, rhs->data.c_arr);

      case mod_json_type_string:
        return (mod_json_string_compare(lhs->data.c_str, rhs->data.c_str) == 0);

      case mod_json_type_integer:
        return (lhs->data.c_int == rhs->data.c_int);

      case mod_json_type_float:
        return mod_json_value_is_equal_float(lhs->data.c_float,
                                             rhs->data.c_float);

      case mod_json_type_boolean:
        return ((!lhs->data.c_bool) == (!rhs->data.c_bool));

      default:
        break;
    }
  }
  return MOD_JSON_FALSE;
}

void mod_json_value_unset(mod_json_value_t *val) {
  if (val && mod_json_value_put(val) <= 0) {
    mod_json_value_clear(val);
    mod_json_free(val);
  }
}

static inline int mod_json_string_expand(mod_json_string_t *str,
                                         mod_json_size_t size) {
  mod_json_char_t *cstr;
  mod_json_size_t len;

  size = mod_json_utils_clp2(size);
  if (size < MOD_JSON_STRING_DEFSIZE) {
    size = MOD_JSON_STRING_DEFSIZE;
  }
  mod_json_minus_if_false(size > str->size);

  cstr = (mod_json_char_t *)mod_json_malloc(size * sizeof(mod_json_char_t));
  mod_json_minus_if_false(cstr);

  len = (mod_json_size_t)(str->last - str->first);
  if (len != 0) {
    memcpy(cstr, str->first, len + 1);
  } else {
    *cstr = '\0'; /* terminal character */
  }
  mod_json_free(str->first);
  str->first = cstr;
  str->last = cstr + len;
  str->size = size;

  /* success */
  return 0;
}

int mod_json_string_reserve(mod_json_string_t *str, mod_json_size_t n) {
  mod_json_minus_if_false(str);

  if (str->size >= n + 1) {
    /* needn't grow */
    return 0;
  }
  return mod_json_string_expand(str, n + 1);
}

static inline mod_json_string_t *mod_json_string_malloc(mod_json_size_t size) {
  mod_json_string_t *str;
  mod_json_char_t *buf;

  buf = (mod_json_char_t *)mod_json_malloc(size * sizeof(mod_json_char_t));
  mod_json_null_if_false(buf);

  str = (mod_json_string_t *)mod_json_malloc(sizeof(mod_json_string_t));
  if (mod_json_unlikely(!str)) {
    mod_json_free(buf);
    return NULL;
  }

  str->refer = 1;
  str->size = size;
  str->first = buf;
  str->last = buf;
  *buf = '\0';
  return str;
}

int mod_json_string_assign(mod_json_string_t *str, mod_json_cchar_t *cstr,
                           mod_json_size_t len) {
  mod_json_string_reset(str);
  mod_json_minus_if_ne_zero(mod_json_string_reserve(str, len));

  if (cstr && len) {
    memcpy(str->first, cstr, len);
  }
  str->last = str->first + len;
  *(str->last) = '\0';

  /* success */
  return 0;
}

static inline mod_json_string_t *mod_json_string_set_empty(void) {
  return mod_json_string_malloc(MOD_JSON_STRING_DEFSIZE);
}

static inline mod_json_string_t *mod_json_string_set_cstr(
    mod_json_cchar_t *cstr, mod_json_size_t len) {
  mod_json_string_t *str;

  str = mod_json_string_malloc(mod_json_utils_clp2(len + 1));
  mod_json_null_if_false(str);

  str->last = str->first + len;
  memcpy(str->first, cstr, len);
  *(str->last) = '\0';
  return str;
}

mod_json_string_t *mod_json_string_set(mod_json_cchar_t *cstr,
                                       mod_json_size_t len) {
  return ((cstr && len) ? mod_json_string_set_cstr(cstr, len)
                        : mod_json_string_set_empty());
}

void mod_json_string_unset(mod_json_string_t *str) {
  if (str && mod_json_string_put(str) <= 0) {
    mod_json_free(str->first);
    mod_json_free(str);
  }
}

void mod_json_string_reset(mod_json_string_t *str) {
  if (str) {
    str->last = str->first;
    *(str->first) = '\0';
  }
}

static inline int mod_json_string_add_char(mod_json_string_t *str,
                                           mod_json_char_t ch) {
  mod_json_size_t need;

  need = (mod_json_size_t)(str->last - str->first) + 2;
  if (need > str->size) {
    mod_json_minus_if_ne_zero(mod_json_string_expand(str, need));
  }

  /* append to string */
  *(str->last++) = ch;
  *(str->last) = '\0';

  /* success */
  return 0;
}

static inline int mod_json_string_add_cstr(mod_json_string_t *str,
                                           mod_json_cchar_t *cstr,
                                           mod_json_size_t len) {
  if (cstr && len) {
    mod_json_size_t need;

    need = len + (mod_json_size_t)(str->last - str->first) + 1;
    if (need > str->size) {
      mod_json_minus_if_ne_zero(mod_json_string_expand(str, need));
    }

    /* append to string */
    memcpy(str->last, cstr, len);
    str->last += len;
    *(str->last) = '\0';
  }

  /* success */
  return 0;
}

static inline int mod_json_string_add_jstr(mod_json_string_t *str,
                                           mod_json_string_t *val) {
  return mod_json_string_add_cstr(str, val->first,
                                  (mod_json_size_t)(val->last - val->first));
}

int mod_json_string_add(mod_json_string_t *str, mod_json_string_t *val) {
  return mod_json_string_add_jstr(str, val);
}

int mod_json_string_append(mod_json_string_t *str, mod_json_cchar_t *cstr,
                           mod_json_size_t len) {
  return mod_json_string_add_cstr(str, cstr, len);
}

mod_json_size_t mod_json_string_hash(mod_json_string_t *str) {
  mod_json_size_t hash = 1;

  if (str) {
    mod_json_cchar_t *iter = str->first;
    mod_json_cchar_t *last = str->last;

    for (; iter != last; ++iter) {
      mod_json_size_t c = (mod_json_size_t)(*iter);
      hash = hash * 131 + c;
    }
  }
  return hash;
}

int mod_json_string_compare(mod_json_string_t *str1, mod_json_string_t *str2) {
  mod_json_size_t len1 = 0, len2 = 0;

  if (str1 == str2) {
    /* The same pointer */
    return 0;
  }

  if (str1) {
    len1 = (mod_json_size_t)(str1->last - str1->first);
    if (str2) {
      len2 = (mod_json_size_t)(str2->last - str2->first);
      if (len1 == len2) {
        return memcmp(str1->first, str2->first, len1);
      }
    }
  } else {
    /* The first string is null, and the second string it not null. */
    len2 = (mod_json_size_t)(str2->last - str2->first);
  }
  return (int)(len1 - len2);
}

mod_json_integer_t mod_json_string_integer(mod_json_string_t *str) {
  return (str ? (mod_json_integer_t)mod_json_utils_strtoi(str->first, NULL, 0)
              : 0);
}

mod_json_float_t mod_json_string_float(mod_json_string_t *str) {
  return (str ? mod_json_utils_strtof(str->first, NULL) : 0.0);
}

static inline int mod_json_string_flat(mod_json_string_t *dst,
                                       mod_json_string_t *src) {
  static mod_json_cchar_t *flattab[32] = {
      "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005",
      "\\u0006", "\\u0007", "\\b",     "\\t",     "\\n",     "\\u000b",
      "\\f",     "\\r",     "\\u000e", "\\u000f", "\\u0010", "\\u0011",
      "\\u0012", "\\u0013", "\\u0014", "\\u0015", "\\u0016", "\\u0017",
      "\\u0018", "\\u0019", "\\u001a", "\\u001b", "\\u001c", "\\u001d",
      "\\u001e", "\\u001f"};

  /* length of items in flat table */
  static const mod_json_uchar_t flatlen[32] = {6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2,
                                               6, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6,
                                               6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

  mod_json_cchar_t *first = src->first;
  mod_json_cchar_t *iter = src->first;
  mod_json_cchar_t *last = src->last;

  /* the whole string */
  for (; iter != last; ++iter) {
    int c = *iter;

    if ((mod_json_uchar_t)c <= 0x1f) {
      if (iter > first) {
        mod_json_minus_if_ne_zero(mod_json_string_add_cstr(
            dst, first, (mod_json_size_t)(iter - first)));
      }
      mod_json_minus_if_ne_zero(
          mod_json_string_add_cstr(dst, flattab[c], flatlen[c]));

      /* skip current character */
      first = iter + 1;
    } else if (c == '\"' || c == '\\') {
      if (iter > first) {
        mod_json_minus_if_ne_zero(mod_json_string_add_cstr(
            dst, first, (mod_json_size_t)(iter - first)));
      }
      mod_json_minus_if_ne_zero(mod_json_string_add_char(dst, '\\'));

      /* don't skip current character */
      first = iter;
    }
  }

  if (iter > first) {
    mod_json_minus_if_ne_zero(
        mod_json_string_add_cstr(dst, first, (mod_json_size_t)(iter - first)));
  }

  /* success */
  return 0;
}

static inline int mod_json_string_unflat(mod_json_string_t *dst,
                                         mod_json_string_t *src) {
  enum {
    state_normal,
    state_rev_slash,
    state_digit_1,
    state_digit_2,
    state_digit_3,
    state_digit_4
  } state;

  mod_json_char_t *pbuf = dst->first;
  mod_json_char_t *pend = dst->first + dst->size;
  mod_json_cchar_t *iter = src->first;
  mod_json_cchar_t *last = src->last;
  mod_json_uchar_t high = 0;
  mod_json_uchar_t low = 0;

  /* the whole string */
  for (state = state_normal; iter != last; ++iter) {
    int c = *iter;

    switch (state) {
      case state_normal:
        if (c != '\\') {
          mod_json_minus_if_false(pbuf < pend);
          *pbuf++ = (mod_json_char_t)c;
        } else {
          /* '\\' in process */
          state = state_rev_slash;
        }
        break;

      case state_rev_slash:
        mod_json_minus_if_false(pbuf < pend);

        switch (c) {
          case '\"':
            state = state_normal;
            *pbuf++ = '\"';
            break;
          case '/':
            state = state_normal;
            *pbuf++ = '/';
            break;
          case 'b':
            state = state_normal;
            *pbuf++ = '\b';
            break;
          case 'f':
            state = state_normal;
            *pbuf++ = '\f';
            break;
          case '\\':
            state = state_normal;
            *pbuf++ = '\\';
            break;
          case 'n':
            state = state_normal;
            *pbuf++ = '\n';
            break;
          case 'r':
            state = state_normal;
            *pbuf++ = '\r';
            break;
          case 't':
            state = state_normal;
            *pbuf++ = '\t';
            break;
          case 'u':
            state = state_digit_1;
            break;
          default:
            return -1;
        }
        break;

      case state_digit_1:
        if ((c = mod_json_utils_char2hex((mod_json_char_t)c)) > 15) {
          /* invalid character */
          return -1;
        }
        high = (mod_json_uchar_t)(c << 4);
        state = state_digit_2;
        break;

      case state_digit_2:
        if ((c = mod_json_utils_char2hex((mod_json_char_t)c)) > 15) {
          /* invalid character */
          return -1;
        }
        high |= (mod_json_uchar_t)c;
        state = state_digit_3;
        break;

      case state_digit_3:
        if ((c = mod_json_utils_char2hex((mod_json_char_t)c)) > 15) {
          /* invalid character */
          return -1;
        }
        low = (mod_json_uchar_t)(c << 4);
        state = state_digit_4;
        break;

      case state_digit_4:
        if ((c = mod_json_utils_char2hex((mod_json_char_t)c)) > 15) {
          /* invalid character */
          return -1;
        }
        low |= (mod_json_uchar_t)c;

        /* decode as a UTF-8 string */
        pbuf = mod_json_utils_uni2utf8(pbuf, (mod_json_size_t)(pend - pbuf),
                                       high, low);
        if (!pbuf) {
          /* lack of buffer */
          return -1;
        }
        state = state_normal;
        break;
    }
  }

  if (state != state_normal) {
    /* uncompleted state */
    return -1;
  }
  mod_json_minus_if_false(pbuf < pend);

  /* update the last pointer */
  *(dst->last = pbuf) = '\0';

  /* success */
  return 0;
}

mod_json_string_t *mod_json_string_encode(mod_json_string_t *src) {
  mod_json_string_t *dst;
  mod_json_null_if_false(src);

  dst = mod_json_string_malloc(
      mod_json_utils_clp2((mod_json_size_t)(src->last - src->first) + 1));
  mod_json_null_if_false(dst);

  if (mod_json_unlikely(mod_json_string_flat(dst, src) != 0)) {
    mod_json_string_unset(dst);
    return NULL;
  }
  return dst;
}

mod_json_string_t *mod_json_string_decode(mod_json_string_t *src) {
  mod_json_string_t *dst;
  mod_json_null_if_false(src);

  dst = mod_json_string_malloc(
      mod_json_utils_clp2((mod_json_size_t)(src->last - src->first) + 1));
  mod_json_null_if_false(dst);

  if (mod_json_unlikely(mod_json_string_unflat(dst, src) != 0)) {
    mod_json_string_unset(dst);
    return NULL;
  }
  return dst;
}

mod_json_array_t *mod_json_array_set(mod_json_size_t size) {
  mod_json_array_t *arr;
  mod_json_value_t **buf;

  size = (size ? mod_json_utils_clp2(size) : MOD_JSON_ARRAY_DEFSIZE);
  buf = (mod_json_value_t **)mod_json_malloc(size * sizeof(mod_json_value_t *));
  mod_json_null_if_false(buf);

  /* create an array */
  arr = (mod_json_array_t *)mod_json_malloc(sizeof(mod_json_array_t));
  if (mod_json_unlikely(!arr)) {
    mod_json_free(buf);
    return NULL;
  }

  arr->refer = 1;
  arr->size = size;
  arr->first = buf;
  arr->last = buf;
  return arr;
}

mod_json_array_t *mod_json_array_clone(mod_json_array_t *arr) {
  mod_json_array_t *arr2 = NULL;

  if (arr) {
    arr2 = mod_json_array_set((mod_json_size_t)(arr->last - arr->first));
    if (arr2) {
      mod_json_value_t **iter = arr->first;

      /* clone items */
      for (; iter != arr->last; ++iter) {
        *arr2->last++ = *iter ? mod_json_value_grab(*iter) : NULL;
      }
    }
  }
  return arr2;
}

mod_json_boolean_t mod_json_array_is_equal(mod_json_array_t *lhs,
                                           mod_json_array_t *rhs) {
  mod_json_value_t **itl, **itr;

  if (lhs == rhs) {
    return MOD_JSON_TRUE;
  }

  if (!lhs || !rhs || ((lhs->last - lhs->first) != (rhs->last - rhs->first))) {
    return MOD_JSON_FALSE;
  }

  /* compare items */
  for (itl = lhs->first, itr = rhs->first; itl != lhs->last; ++itl, ++itr) {
    if (!mod_json_value_is_equal(*itl, *itr)) {
      return MOD_JSON_FALSE;
    }
  }
  return MOD_JSON_TRUE;
}

void mod_json_array_unset(mod_json_array_t *arr) {
  if (arr && mod_json_array_put(arr) <= 0) {
    mod_json_value_t **iter = arr->first;

    for (; iter != arr->last; ++iter) {
      mod_json_value_unset(*iter);
    }
    mod_json_free(arr->first);
    mod_json_free(arr);
  }
}

void mod_json_array_reset(mod_json_array_t *arr) {
  if (arr) {
    mod_json_value_t **iter = arr->first;

    for (; iter != arr->last; ++iter) {
      mod_json_value_unset(*iter);
    }
    arr->last = arr->first;
  }
}

static inline void mod_json_array_migrate(mod_json_array_t *arr,
                                          mod_json_value_t **buf,
                                          mod_json_size_t size) {
  mod_json_size_t count = (mod_json_size_t)(arr->last - arr->first);
  if (count > 0) {
    memcpy(buf, arr->first, count * sizeof(mod_json_value_t *));
  }
  mod_json_free(arr->first);

  arr->first = buf;
  arr->last = buf + count;
  arr->size = size;
}

static inline int mod_json_array_expand(mod_json_array_t *arr,
                                        mod_json_size_t n) {
  mod_json_size_t size;
  mod_json_value_t **vals;

  size = mod_json_utils_clp2(n);
  if (size < MOD_JSON_ARRAY_DEFSIZE) {
    size = MOD_JSON_ARRAY_DEFSIZE;
  }
  mod_json_minus_if_false(size > arr->size);

  vals =
      (mod_json_value_t **)mod_json_malloc(size * sizeof(mod_json_value_t *));
  mod_json_minus_if_false(vals);

  /* use new buffer */
  mod_json_array_migrate(arr, vals, size);

  /* success */
  return 0;
}

int mod_json_array_reserve(mod_json_array_t *arr, mod_json_size_t n) {
  mod_json_minus_if_false(arr);

  if (arr->size >= n) {
    /* needn't grow */
    return 0;
  }
  return mod_json_array_expand(arr, n);
}

void mod_json_array_reverse(mod_json_array_t *arr) {
  if (arr) {
    mod_json_value_t **first = arr->first;
    mod_json_value_t **last = arr->last - 1;

    while (first < last) {
      mod_json_value_t *temp = *first;
      *first++ = *last;
      *last-- = temp;
    }
  }
}

int mod_json_array_push(mod_json_array_t *arr, mod_json_value_t *val) {
  mod_json_size_t count;
  mod_json_minus_if_false(arr);

  count = (mod_json_size_t)(arr->last - arr->first);
  if (count >= arr->size) {
    mod_json_minus_if_ne_zero(mod_json_array_expand(arr, count + 1));
  }

  *arr->last++ = val ? mod_json_value_grab(val) : NULL;
  return 0;
}

void mod_json_array_pop(mod_json_array_t *arr) {
  if (arr && arr->first != arr->last) {
    mod_json_value_unset(*(--arr->last));
  }
}

void mod_json_array_shift(mod_json_array_t *arr) {
  if (arr && arr->first != arr->last) {
    mod_json_value_t **it = arr->first;
    mod_json_value_t **last = --arr->last;

    mod_json_value_unset(*it++);
    for (; it <= last; ++it) {
      *(it - 1) = *it;
    }
  }
}

mod_json_value_t *mod_json_array_at(mod_json_array_t *arr, mod_json_size_t id) {
  if (arr && ((arr->first + id) < arr->last)) {
    return (arr->first[id]);
  }
  return NULL;
}

int mod_json_array_merge(mod_json_array_t *dst, mod_json_array_t *src) {
  long count, len1, len2;

  mod_json_minus_if_false(dst && src && dst != src);

  /* update length of array */
  len1 = (mod_json_size_t)(src->last - src->first);
  len2 = (mod_json_size_t)(dst->last - dst->first);
  mod_json_minus_if_false(len1 >= 0 && len2 >= 0);

  /* append empty values */
  count = len1 - len2;
  for (; count > 0; --count) {
    mod_json_array_push(dst, NULL);
  }

  /* It must be assigned again. */
  len2 = (mod_json_size_t)(dst->last - dst->first);
  count = (len1 < len2 ? len1 : len2);

  while ((count--) > 0) {
    mod_json_value_t **iter1 = src->first + count;
    mod_json_value_t **iter2 = dst->first + count;

    if (!(*iter2)) {
      *iter2 = *iter1 ? mod_json_value_grab(*iter1) : NULL;
      continue;
    }

    if (mod_json_value_is_shared(*iter2)) {
      mod_json_value_put(*iter2);
      *iter2 = mod_json_value_clone(*iter2);
    }
    mod_json_value_merge(*iter2, *iter1);
  }

  /* success */
  return 0;
}

int mod_json_array_resize(mod_json_array_t *arr, mod_json_size_t n,
                          mod_json_value_t *val) {
  mod_json_size_t orig;

  /* check input */
  mod_json_minus_if_false(arr);

  /* original count of array */
  orig = (mod_json_size_t)(arr->last - arr->first);

  if (orig < n) {
    mod_json_value_t **iter;

    if (arr->size < n) {
      mod_json_minus_if_ne_zero(mod_json_array_expand(arr, n));
    }

    iter = arr->last;
    arr->last = arr->first + n;

    /* grab the first one, but get the others */
    *iter++ = val = val ? mod_json_value_grab(val) : NULL;
    for (; iter != arr->last; ++iter) {
      *iter = val ? mod_json_value_get(val) : NULL;
    }
  } else if (orig > n) {
    mod_json_value_t **iter = arr->first + n;

    for (; iter != arr->last; ++iter) {
      mod_json_value_unset(*iter);
      *iter = NULL;
    }
    arr->last = arr->first + n;
  }

  /* success */
  return 0;
}

static inline void mod_json_pair_init(mod_json_pair_t *pair,
                                      mod_json_string_t *key,
                                      mod_json_value_t *val) {
  pair->key = mod_json_string_grab(key);
  pair->val = val ? mod_json_value_grab(val) : NULL;
}

static inline void mod_json_pair_cleanup(mod_json_pair_t *pair) {
  mod_json_string_unset(pair->key);
  mod_json_value_unset(pair->val);
  pair->key = NULL;
  pair->val = NULL;
}

mod_json_object_t *mod_json_object_set(mod_json_size_t size) {
  mod_json_object_t *obj;
  mod_json_pair_t *buf;

  size = (size ? mod_json_utils_clp2(size) : MOD_JSON_OBJECT_DEFSIZE);
  buf = (mod_json_pair_t *)mod_json_malloc(size * sizeof(mod_json_pair_t));
  mod_json_null_if_false(buf);

  /* create a object */
  obj = (mod_json_object_t *)mod_json_malloc(sizeof(mod_json_object_t));
  if (mod_json_unlikely(!obj)) {
    mod_json_free(buf);
    return NULL;
  }

  obj->refer = 1;
  obj->size = size;
  obj->first = buf;
  obj->last = buf;
  return obj;
}

void mod_json_object_unset(mod_json_object_t *obj) {
  if (obj && mod_json_object_put(obj) <= 0) {
    mod_json_pair_t *iter = obj->first;

    for (; iter != obj->last; ++iter) {
      mod_json_pair_cleanup(iter);
    }
    mod_json_free(obj->first);
    mod_json_free(obj);
  }
}

void mod_json_object_reset(mod_json_object_t *obj) {
  if (obj) {
    mod_json_pair_t *iter = obj->first;

    for (; iter != obj->last; ++iter) {
      mod_json_pair_cleanup(iter);
    }
    obj->last = obj->first;
  }
}

static inline void mod_json_object_migrate(mod_json_object_t *obj,
                                           mod_json_pair_t *buf,
                                           mod_json_size_t size) {
  mod_json_size_t count = (mod_json_size_t)(obj->last - obj->first);
  if (count > 0) {
    memcpy(buf, obj->first, count * sizeof(mod_json_pair_t));
  }
  mod_json_free(obj->first);

  obj->first = buf;
  obj->last = buf + count;
  obj->size = size;
}

static inline int mod_json_object_expand(mod_json_object_t *obj,
                                         mod_json_size_t n) {
  mod_json_size_t size;
  mod_json_pair_t *buf;

  size = mod_json_utils_clp2(n);
  if (size < MOD_JSON_OBJECT_DEFSIZE) {
    size = MOD_JSON_OBJECT_DEFSIZE;
  }
  mod_json_minus_if_false(size > obj->size);

  buf = (mod_json_pair_t *)mod_json_malloc(size * sizeof(mod_json_pair_t));
  mod_json_minus_if_false(buf);

  /* use new buffer */
  mod_json_object_migrate(obj, buf, size);

  /* success */
  return 0;
}

static inline mod_json_pair_t *mod_json_object_find_pair(mod_json_object_t *obj,
                                                         mod_json_string_t *key,
                                                         mod_json_size_t *out) {
  mod_json_pair_t *first = obj->first;
  mod_json_pair_t *last = obj->last;

  while (first < last) {
    mod_json_pair_t *middle = first + ((last - first) >> 2);
    int diff = mod_json_string_compare(middle->key, key);

    if (diff < 0) {
      first = middle + 1;
    } else if (diff > 0) {
      last = middle;
    } else /*if (diff == 0)*/
    {
      *out = (mod_json_size_t)(middle - obj->first);
      return middle;
    }
  }
  *out = (mod_json_size_t)(first - obj->first);
  return NULL;
}

mod_json_pair_t *mod_json_object_insert_force(mod_json_object_t *obj,
                                              mod_json_size_t npos,
                                              mod_json_string_t *key,
                                              mod_json_value_t *val) {
  mod_json_pair_t *iter, *pos;
  mod_json_size_t count;

  count = (mod_json_size_t)(obj->last - obj->first);
  if (count >= obj->size) {
    mod_json_null_if_ne_zero(mod_json_object_expand(obj, count + 1));
  }

  pos = obj->first + npos;
  iter = obj->last++;
  for (; iter != pos; --iter) {
    mod_json_pair_t *prev = iter - 1;
    iter->key = prev->key;
    iter->val = prev->val;
  }
  mod_json_pair_init(pos, key, val);
  return pos;
}

mod_json_pair_t *mod_json_object_insert(mod_json_object_t *obj,
                                        mod_json_string_t *key,
                                        mod_json_value_t *val) {
  mod_json_size_t npos;
  mod_json_null_if_false(obj && key);

  if (mod_json_object_find_pair(obj, key, &npos)) {
    /* One in object */
    return NULL;
  }
  return mod_json_object_insert_force(obj, npos, key, val);
}

mod_json_pair_t *mod_json_object_assign(mod_json_object_t *obj,
                                        mod_json_string_t *key,
                                        mod_json_value_t *val) {
  mod_json_pair_t *elem = NULL;

  if (obj && key) {
    mod_json_size_t npos;

    elem = mod_json_object_find_pair(obj, key, &npos);
    if (elem) {
      if (!elem->val) {
        elem->val = val ? mod_json_value_grab(val) : NULL;
      } else {
        /* overwrite the old value */
        mod_json_value_assign(elem->val, val);
      }
    } else {
      /* insert a new one */
      elem = mod_json_object_insert_force(obj, npos, key, val);
    }
  }
  return elem;
}

mod_json_pair_t *mod_json_object_touch(mod_json_object_t *obj,
                                       mod_json_cchar_t *key) {
  mod_json_pair_t *elem = NULL;

  if (obj && key) {
    mod_json_string_t str;
    mod_json_size_t npos;

    str.first = (mod_json_char_t *)key;
    str.last = str.first + mod_json_utils_strlen(key);

    elem = mod_json_object_find_pair(obj, &str, &npos);
    if (!elem) {
      mod_json_string_t *jkey;

      /* insert a new one */
      jkey =
          mod_json_string_set(key, (mod_json_size_t)mod_json_utils_strlen(key));
      elem = mod_json_object_insert_force(obj, npos, jkey, NULL);
      mod_json_string_unset(jkey);
    }
  }
  return elem;
}

mod_json_object_t *mod_json_object_clone(mod_json_object_t *obj) {
  mod_json_object_t *obj2 = NULL;

  if (obj) {
    obj2 = mod_json_object_set((mod_json_size_t)(obj->last - obj->first));
    if (obj2) {
      mod_json_pair_t *iter = obj->first;

      /* clone items */
      for (; iter != obj->last; ++iter) {
        mod_json_pair_init(obj2->last++, iter->key, iter->val);
      }
    }
  }
  return obj2;
}

mod_json_boolean_t mod_json_object_is_equal(mod_json_object_t *lhs,
                                            mod_json_object_t *rhs) {
  mod_json_pair_t *itl, *itr;

  if (lhs == rhs) {
    /* The same pointer */
    return MOD_JSON_TRUE;
  }

  if (!lhs || !rhs || ((lhs->last - lhs->first) != (rhs->last - rhs->first))) {
    return MOD_JSON_FALSE;
  }

  /* compare items */
  for (itl = lhs->first, itr = rhs->first; itl != lhs->last; ++itl, ++itr) {
    if ((mod_json_string_compare(itl->key, itr->key) != 0) ||
        (!mod_json_value_is_equal(itl->val, itr->val))) {
      return MOD_JSON_FALSE;
    }
  }
  return MOD_JSON_TRUE;
}

void mod_json_object_erase(mod_json_object_t *obj, mod_json_cchar_t *key) {
  if (obj && key) {
    mod_json_string_t str;
    mod_json_pair_t *iter;
    mod_json_size_t npos;

    str.first = (mod_json_char_t *)key;
    str.last = str.first + mod_json_utils_strlen(key);

    iter = mod_json_object_find_pair(obj, &str, &npos);
    if (iter) {
      mod_json_pair_cleanup(iter++);

      for (; iter != obj->last; ++iter) {
        mod_json_pair_t *prev = iter - 1;
        prev->key = iter->key;
        prev->val = iter->val;
      }
      --obj->last;
    }
  }
}

mod_json_value_t *mod_json_object_at(mod_json_object_t *obj,
                                     mod_json_cchar_t *key) {
  if (obj && key) {
    mod_json_string_t str;
    mod_json_pair_t *elem;
    mod_json_size_t npos;

    str.first = (mod_json_char_t *)key;
    str.last = str.first + mod_json_utils_strlen(key);

    elem = mod_json_object_find_pair(obj, &str, &npos);
    if (elem) {
      return (elem->val);
    }
  }
  return NULL;
}

mod_json_pair_t *mod_json_object_find(mod_json_object_t *obj,
                                      mod_json_cchar_t *key) {
  if (obj && key) {
    mod_json_string_t str;
    mod_json_size_t npos;

    str.first = (mod_json_char_t *)key;
    str.last = str.first + mod_json_utils_strlen(key);

    return mod_json_object_find_pair(obj, &str, &npos);
  }
  return NULL;
}

int mod_json_object_merge(mod_json_object_t *dst, mod_json_object_t *src) {
  mod_json_pair_t *iter;

  mod_json_minus_if_false(dst && src && dst != src);

  for (iter = src->first; iter != src->last; ++iter) {
    mod_json_pair_t *elem;
    mod_json_size_t npos;

    elem = mod_json_object_find_pair(dst, iter->key, &npos);
    if (!elem) {
      /* insert a new one */
      mod_json_object_insert_force(dst, npos, iter->key, iter->val);
      continue;
    }

    if (!elem->val) {
      elem->val = iter->val ? mod_json_value_grab(iter->val) : NULL;
      continue;
    }

    if (mod_json_value_is_shared(elem->val)) {
      mod_json_value_put(elem->val);
      elem->val = mod_json_value_clone(elem->val);
    }
    mod_json_value_merge(elem->val, iter->val);
  }
  return 0;
}

static inline mod_json_cchar_t *mod_json_token_strskp(mod_json_token_t *tok,
                                                      mod_json_cchar_t *cstr) {
  if ((tok->options & MOD_JSON_COMMENT) == 0) {
    return mod_json_utils_strskpb(cstr);
  }
  return mod_json_utils_strskp(cstr);
}

static inline mod_json_cchar_t *mod_json_token_strfquo(mod_json_token_t *tok,
                                                       mod_json_cchar_t *cstr,
                                                       mod_json_char_t quo) {
  if ((tok->options & MOD_JSON_UNSTRICT) == 0) {
    return mod_json_utils_strfquo(cstr, quo);
  }
  return mod_json_utils_strfquo2(cstr, quo);
}

static inline mod_json_cchar_t *mod_json_token_strfsep(mod_json_token_t *tok,
                                                       mod_json_cchar_t *cstr) {
  if ((tok->options & MOD_JSON_COMMENT) == 0) {
    return mod_json_utils_strfsep(cstr);
  }
  return mod_json_utils_strfsep2(cstr);
}

mod_json_token_t *mod_json_token_create(mod_json_option_t *opt) {
  mod_json_token_t *tok;
  mod_json_size_t opts = MOD_JSON_TOKEN_DEFOPTS;
  mod_json_size_t mobj = MOD_JSON_TOKEN_DEFOBJDEP;
  mod_json_size_t marr = MOD_JSON_TOKEN_DEFARRDEP;

  if (opt) {
    opts = opt->options;

    if (opt->object_depth > 0) {
      mobj = opt->object_depth;
    }
    if (opt->array_depth > 0) {
      marr = opt->array_depth;
    }
  }

  tok = (mod_json_token_t *)mod_json_malloc(
      (mobj + marr) * sizeof(mod_json_char_t) + sizeof(mod_json_token_t));
  mod_json_null_if_false(tok);

  memset(tok, 0, sizeof(mod_json_token_t));
  tok->state = mod_json_state_null;
  tok->error = mod_json_error_null;
  tok->options = opts;
  tok->object_max_depth = mobj;
  tok->array_max_depth = marr;
  return tok;
}

void mod_json_token_destroy(mod_json_token_t *tok) {
  mod_json_free(tok);
}

static inline void mod_json_token_set_tag(mod_json_token_t *tok,
                                          mod_json_char_t tag) {
  mod_json_size_t depth = tok->object_depth + tok->array_depth;
  if (depth != 0) {
    tok->tags[depth - 1] = tag;
  }
}

static inline mod_json_char_t mod_json_token_tag(mod_json_token_t *tok) {
  mod_json_size_t depth = tok->object_depth + tok->array_depth;

  /* type of current depth */
  return (depth ? tok->tags[depth - 1] : (mod_json_char_t)-1);
}

mod_json_error_t mod_json_token_error(mod_json_token_t *tok) {
  return (tok->error);
}

mod_json_cchar_t *mod_json_token_context(mod_json_token_t *tok) {
  return (tok->context);
}

mod_json_state_t mod_json_token_state(mod_json_token_t *tok) {
  return (tok->state);
}

mod_json_size_t mod_json_token_object_depth(mod_json_token_t *tok) {
  return (tok->object_depth);
}

mod_json_size_t mod_json_token_array_depth(mod_json_token_t *tok) {
  return (tok->array_depth);
}

mod_json_size_t mod_json_token_depth(mod_json_token_t *tok) {
  return (tok->object_depth + tok->array_depth);
}

mod_json_size_t mod_json_token_max_object_depth(mod_json_token_t *tok) {
  return (tok->object_max_depth);
}

mod_json_size_t mod_json_token_max_array_depth(mod_json_token_t *tok) {
  return (tok->array_max_depth);
}

mod_json_size_t mod_json_token_max_depth(mod_json_token_t *tok) {
  return (tok->object_max_depth + tok->array_max_depth);
}

mod_json_void_t *mod_json_token_param(mod_json_token_t *tok) {
  return (tok->param);
}

void mod_json_token_set_param(mod_json_token_t *tok, mod_json_void_t *param) {
  tok->param = param;
}

void mod_json_token_set_event(mod_json_token_t *tok, mod_json_event_proc proc) {
  tok->event_proc = proc;
}

mod_json_event_t mod_json_token_event(mod_json_token_t *tok) {
  return (tok->event_code);
}

static inline int mod_json_token_invoke_field(mod_json_token_t *tok,
                                              mod_json_cchar_t *val,
                                              mod_json_size_t len) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_field;
    return invoke(tok, (mod_json_void_t *)val, len);
  }
  return 0;
}

static inline int mod_json_token_invoke_object(mod_json_token_t *tok) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_object;
    return invoke(tok, NULL, 0);
  }
  return 0;
}

static inline int mod_json_token_invoke_array(mod_json_token_t *tok) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_array;
    return invoke(tok, NULL, 0);
  }
  return 0;
}

static inline int mod_json_token_invoke_null(mod_json_token_t *tok) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_null;
    return invoke(tok, NULL, 0);
  }
  return 0;
}

static inline int mod_json_token_invoke_boolean(mod_json_token_t *tok,
                                                mod_json_boolean_t val) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_boolean;
    return invoke(tok, &val, sizeof(val));
  }
  return 0;
}

static inline int mod_json_token_invoke_integer(mod_json_token_t *tok,
                                                mod_json_integer_t val) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_integer;
    return invoke(tok, &val, sizeof(val));
  }
  return 0;
}

static inline int mod_json_token_invoke_float(mod_json_token_t *tok,
                                              mod_json_float_t val) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_float;
    return invoke(tok, &val, sizeof(val));
  }
  return 0;
}

static inline int mod_json_token_invoke_string(mod_json_token_t *tok,
                                               mod_json_cchar_t *val,
                                               mod_json_size_t len) {
  mod_json_event_proc invoke = tok->event_proc;
  if (invoke) {
    tok->event_code = mod_json_event_string;
    return invoke(tok, (mod_json_void_t *)val, len);
  }
  return 0;
}

static inline mod_json_cchar_t *mod_json_token_start(mod_json_token_t *tok,
                                                     mod_json_cchar_t *cstr) {
  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case '{':
      tok->state = mod_json_state_object_start;
      return (cstr + 1);

    case '[':
      tok->state = mod_json_state_array_start;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_empty;
      tok->context = cstr;
      break;

    default:
      tok->error = mod_json_error_start;
      tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_value_null(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  mod_json_char_t c1 = *(cstr + 1);
  mod_json_char_t c2 = *(cstr + 2);
  mod_json_char_t c3 = *(cstr + 3);

  if ((c1 != 'u' && c1 != 'U') || (c2 != 'l' && c2 != 'L') ||
      (c3 != 'l' && c3 != 'L')) {
    tok->error = mod_json_error_value;
    tok->context = cstr;
    return NULL;
  }

  if (mod_json_token_invoke_null(tok) != 0) {
    tok->error = mod_json_error_break;
    tok->context = cstr;
    return NULL;
  }
  return (cstr + 4);
}

static inline mod_json_cchar_t *mod_json_token_value_true(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  mod_json_char_t c1 = *(cstr + 1);
  mod_json_char_t c2 = *(cstr + 2);
  mod_json_char_t c3 = *(cstr + 3);

  if ((c1 != 'r' && c1 != 'R') || (c2 != 'u' && c2 != 'U') ||
      (c3 != 'e' && c3 != 'E')) {
    tok->error = mod_json_error_value;
    tok->context = cstr;
    return NULL;
  }

  if (mod_json_token_invoke_boolean(tok, MOD_JSON_TRUE) != 0) {
    tok->error = mod_json_error_break;
    tok->context = cstr;
    return NULL;
  }
  return (cstr + 4);
}

static inline mod_json_cchar_t *mod_json_token_value_false(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  mod_json_char_t c1 = *(cstr + 1);
  mod_json_char_t c2 = *(cstr + 2);
  mod_json_char_t c3 = *(cstr + 3);
  mod_json_char_t c4 = *(cstr + 4);

  if ((c1 != 'a' && c1 != 'A') || (c2 != 'l' && c2 != 'L') ||
      (c3 != 's' && c3 != 'S') || (c4 != 'e' && c4 != 'E')) {
    tok->error = mod_json_error_value;
    tok->context = cstr;
    return NULL;
  }

  if (mod_json_token_invoke_boolean(tok, MOD_JSON_FALSE) != 0) {
    tok->error = mod_json_error_break;
    tok->context = cstr;
    return NULL;
  }
  return (cstr + 5);
}

static inline mod_json_cchar_t *mod_json_token_value_infinity(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  mod_json_char_t c1 = *(cstr + 1);
  mod_json_char_t c2 = *(cstr + 2);

  if ((c1 != 'n' && c1 != 'N') || (c2 != 'f' && c2 != 'F')) {
    tok->error = mod_json_error_value;
    tok->context = cstr;
    return NULL;
  }

  if (mod_json_token_invoke_float(tok, MOD_JSON_INFINITY) != 0) {
    tok->error = mod_json_error_break;
    tok->context = cstr;
    return NULL;
  }
  return (cstr + 3);
}

static inline mod_json_cchar_t *mod_json_token_value_string(
    mod_json_token_t *tok, mod_json_cchar_t *cstr, mod_json_char_t quo) {
  mod_json_cchar_t *cstr2 = mod_json_token_strfquo(tok, ++cstr, quo);
  if (!cstr2) {
    tok->error = mod_json_error_quote;
    tok->context = cstr;
    return NULL;
  }

  if (mod_json_token_invoke_string(tok, cstr,
                                   (mod_json_size_t)(cstr2 - cstr)) != 0) {
    tok->error = mod_json_error_break;
    tok->context = cstr;
    return NULL;
  }
  return (cstr2 + 1);
}

static inline mod_json_cchar_t *mod_json_token_value_number(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  enum { number_integer, number_float } num_type = number_integer;

  mod_json_float_t dbl = 0.0;
  uint32_t dig = 0;
  uint64_t u64 = 0;
  int32_t minus = 0;
  int32_t exp_frac = 0, exp = 0;

  /* Parse minus */
  minus = *cstr;
  if (minus == '-' || minus == '+') {
    ++cstr;
  }

  /* The first digit */
  if ((dig = (uint32_t)(*cstr - '0')) > 9) {
    return NULL;
  }

  /* Save the first digit */
  u64 = dig;

  /* Parse as 64bit integer */
  if (minus != '-') {
    while ((dig = (uint32_t)(*(++cstr) - '0')) <= 9) {
      if (u64 >= 1844674407370955161uLL) {
        /* 2^64 - 1 = 18446744073709551615 */
        if (u64 != 1844674407370955161uLL || dig > 5) {
          dbl = (mod_json_float_t)u64 * 10 + dig;
          num_type = number_float;
          break;
        }
      }
      u64 = u64 * 10 + dig;
    }
  } else {
    while ((dig = (uint32_t)(*(++cstr) - '0')) <= 9) {
      /* 2^63 = 9223372036854775808 */
      if (u64 >= 922337203685477580uLL) {
        if (u64 != 922337203685477580uLL || dig > 8) {
          dbl = (mod_json_float_t)u64 * 10 + dig;
          num_type = number_float;
          break;
        }
      }
      u64 = u64 * 10 + dig;
    }
  }

  /* Force double for big integer */
  if (num_type == number_float) {
    while ((dig = (uint32_t)(*(++cstr) - '0')) <= 9) {
      if (dbl >= 1E307) {
        /* Number too big to store in double */
        return NULL;
      }
      dbl = dbl * 10 + dig;
    }
  }

  /* Parse frac = decimal-point 1*DIGIT */
  if (*cstr == '.') {
    if (num_type != number_float) {
      dbl = (mod_json_float_t)u64;
      num_type = number_float;
    }

    if ((dig = (uint32_t)(*(++cstr) - '0')) > 9) {
      /* At least one digit in fraction part */
      return NULL;
    }

    dbl = dbl * 10 + dig;
    --exp_frac;

    while ((dig = (uint32_t)(*(++cstr) - '0')) <= 9) {
      if (exp_frac > -16) {
        dbl = dbl * 10 + dig;
        --exp_frac;
      }
    }
  }

  /* Parse exp = e [ minus / plus ] 1*DIGIT */
  if (*cstr == 'e' || *cstr == 'E') {
    int32_t exp_minus = 0;

    if (num_type != number_float) {
      dbl = (mod_json_float_t)u64;
      num_type = number_float;
    }

    exp_minus = *(++cstr);
    if (exp_minus == '-' || exp_minus == '+') {
      ++cstr;
    }

    /* The first number char after 'e/E' */
    if ((dig = (uint32_t)(*cstr - '0')) > 9) {
      return NULL;
    }
    exp = (int32_t)dig;

    while ((dig = (uint32_t)(*(++cstr) - '0')) <= 9) {
      exp = exp * 10 + (int32_t)dig;
      if (exp > 308) {
        /* Number too big to store in double */
        return NULL;
      }
    }

    if (exp_minus == '-') {
      exp = -exp;
    }
  }

  /* Finish parsing, call event according to the type of number. */
  if (num_type == number_float) {
    dbl *= mod_json_utils_pow10(exp + exp_frac);
    if (minus == '-') {
      dbl = -dbl;
    }
    if (mod_json_token_invoke_float(tok, dbl) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }
  } else {
    if (minus == '-') {
      u64 = (uint64_t)(-(int64_t)u64);
    }
    if (mod_json_token_invoke_integer(tok, (mod_json_integer_t)u64) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }
  }
  return cstr;
}

static inline mod_json_cchar_t *mod_json_token_array_start(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  if (tok->array_depth < tok->array_max_depth) {
    /* callback */
    if (mod_json_token_invoke_array(tok) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    /* increase depth */
    ++tok->array_depth;

    /* push current tag */
    mod_json_token_set_tag(tok, '[');

    cstr = mod_json_token_strskp(tok, cstr);
    switch (*cstr) {
      case '[':
        tok->state = mod_json_state_array_start;
        return (cstr + 1);

      case ']':
        tok->state = mod_json_state_array_finish;
        return (cstr + 1);

      case '\0':
        tok->error = mod_json_error_trunc;
        tok->context = cstr;
        break;

      default:
        tok->state = mod_json_state_array_half;
        return (cstr);
    }
  } else {
    tok->error = mod_json_error_depth;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_array_half(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case ',':
      tok->state = mod_json_state_array_half;
      return (cstr + 1);

    case '[':
      tok->state = mod_json_state_array_start;
      return (cstr + 1);

    case ']':
      tok->state = mod_json_state_array_finish;
      return (cstr + 1);

    case '{':
      tok->state = mod_json_state_object_start;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_trunc;
      tok->context = cstr;
      return NULL;

    /* value in array */
    case 't':
    case 'T':
      cstr = mod_json_token_value_true(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case 'f':
    case 'F':
      cstr = mod_json_token_value_false(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case 'n':
    case 'N':
      cstr = mod_json_token_value_null(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '+':
    case '-':
      cstr = mod_json_token_value_number(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case '\"':
      cstr = mod_json_token_value_string(tok, cstr, '\"');
      if (!cstr) {
        return NULL;
      }
      break;

    case '\'':
      if (tok->options & MOD_JSON_SQUOTE) {
        cstr = mod_json_token_value_string(tok, cstr, '\'');
        if (!cstr) {
          return NULL;
        }
        break;
      }
      /* FALLTHRU */

    default:
      tok->error = mod_json_error_value;
      tok->context = cstr;
      return NULL;
  }

  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case ',':
      tok->state = mod_json_state_array_half;
      return (cstr + 1);

    case ']':
      tok->state = mod_json_state_array_finish;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_trunc;
      tok->context = cstr;
      break;

    default:
      tok->error = mod_json_error_value;
      tok->context = cstr;
      break;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_array_finish(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  if (tok->array_depth) {
    /* decrease depth */
    --tok->array_depth;

    /* callback */
    if (mod_json_token_invoke_array(tok) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    cstr = mod_json_token_strskp(tok, cstr);
    switch (*cstr) {
      case ']':
        tok->state = mod_json_state_array_finish;
        return (cstr + 1);

      case '}':
        tok->state = mod_json_state_object_finish;
        return (cstr + 1);

      case '\0':
        if (tok->object_depth || tok->array_depth) {
          tok->error = mod_json_error_trunc;
          tok->context = cstr;
        } else {
          tok->state = mod_json_state_finish;
        }
        break;

      case ',':
        if (tok->object_depth || tok->array_depth) {
          mod_json_char_t tag = mod_json_token_tag(tok);

          if (tag == '{') {
            tok->state = mod_json_state_object_half1;
            return (cstr + 1);
          } else if (tag == '[') {
            tok->state = mod_json_state_array_half;
            return (cstr + 1);
          }
        }
        /* FALLTHRU */

      default:
        tok->error = mod_json_error_array;
        tok->context = cstr;
    }
  } else {
    tok->error = mod_json_error_depth;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_start(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  if (tok->object_depth < tok->object_max_depth) {
    /* callback */
    if (mod_json_token_invoke_object(tok) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    /* increase depth */
    ++tok->object_depth;

    /* push current tag */
    mod_json_token_set_tag(tok, '{');

    cstr = mod_json_token_strskp(tok, cstr);
    switch (*cstr) {
      case '}':
        tok->state = mod_json_state_object_finish;
        return (cstr + 1);

      case '\0':
        tok->error = mod_json_error_trunc;
        tok->context = cstr;
        break;

      default:
        tok->state = mod_json_state_object_half1;
        return (cstr);
    }
  } else {
    tok->error = mod_json_error_depth;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_quotekey(
    mod_json_token_t *tok, mod_json_cchar_t *cstr, mod_json_char_t quo) {
  mod_json_cchar_t *cstr2 = mod_json_token_strfquo(tok, ++cstr, quo);
  if (cstr2) {
    /* callback */
    if (mod_json_token_invoke_field(tok, cstr,
                                    (mod_json_size_t)(cstr2 - cstr)) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    cstr2 = mod_json_token_strskp(tok, ++cstr2);
    switch (*cstr2) {
      case ':':
        tok->state = mod_json_state_object_half2;
        return (cstr2 + 1);

      case '\0':
        tok->error = mod_json_error_trunc;
        tok->context = cstr;
        break;

      default:
        tok->error = mod_json_error_key;
        tok->context = cstr2;
        break;
    }
  } else {
    tok->error = mod_json_error_quote;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_simplekey(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  mod_json_cchar_t *cstr2 = mod_json_token_strfsep(tok, cstr);
  if (cstr2 != cstr) {
    /* callback */
    if (mod_json_token_invoke_field(tok, cstr,
                                    (mod_json_size_t)(cstr2 - cstr)) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    cstr2 = mod_json_token_strskp(tok, cstr2);
    switch (*cstr2) {
      case ':':
        tok->state = mod_json_state_object_half2;
        return (cstr2 + 1);

      case '\0':
        tok->error = mod_json_error_trunc;
        tok->context = cstr;
        break;

      default:
        tok->error = mod_json_error_key;
        tok->context = cstr2;
        break;
    }
  } else {
    tok->error = mod_json_error_key;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_half1(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case ',':
      tok->state = mod_json_state_object_half1;
      return (cstr + 1);

    case '}':
      tok->state = mod_json_state_object_finish;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_trunc;
      tok->context = cstr;
      break;

    case '\"':
      /* The key with double quotes */
      return mod_json_token_object_quotekey(tok, cstr, '\"');

    case '\'':
      if (tok->options & MOD_JSON_SQUOTE) {
        /* The key with single quotes */
        return mod_json_token_object_quotekey(tok, cstr, '\'');
      }
      /* FALLTHRU */

    default:
      /* support simple format? */
      if (tok->options & MOD_JSON_SIMPLE) {
        return mod_json_token_object_simplekey(tok, cstr);
      } else {
        tok->error = mod_json_error_quote;
        tok->context = cstr;
      }
      break;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_half2(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case '{':
      tok->state = mod_json_state_object_start;
      return (cstr + 1);

    case '[':
      tok->state = mod_json_state_array_start;
      return (cstr + 1);

    case ',':
      tok->state = mod_json_state_object_half1;
      return (cstr + 1);

    case '}':
      tok->state = mod_json_state_object_finish;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_trunc;
      tok->context = cstr;
      return NULL;

    case 't':
    case 'T':
      cstr = mod_json_token_value_true(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case 'f':
    case 'F':
      cstr = mod_json_token_value_false(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case 'i':
    case 'I':
      cstr = mod_json_token_value_infinity(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case 'n':
    case 'N':
      cstr = mod_json_token_value_null(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
    case '+':
    case '-':
      cstr = mod_json_token_value_number(tok, cstr);
      if (!cstr) {
        return NULL;
      }
      break;

    case '\"':
      cstr = mod_json_token_value_string(tok, cstr, '\"');
      if (!cstr) {
        return NULL;
      }
      break;

    case '\'':
      if (tok->options & MOD_JSON_SQUOTE) {
        cstr = mod_json_token_value_string(tok, cstr, '\'');
        if (!cstr) {
          return NULL;
        }
        break;
      }
      /* FALLTHRU */

    default:
      tok->error = mod_json_error_value;
      tok->context = cstr;
      return NULL;
  }

  cstr = mod_json_token_strskp(tok, cstr);
  switch (*cstr) {
    case ',':
      tok->state = mod_json_state_object_half1;
      return (cstr + 1);

    case '}':
      tok->state = mod_json_state_object_finish;
      return (cstr + 1);

    case '\0':
      tok->error = mod_json_error_trunc;
      tok->context = cstr;
      break;

    default:
      tok->error = mod_json_error_value;
      tok->context = cstr;
      break;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_object_finish(
    mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  if (tok->object_depth) {
    /* decrease depth */
    --tok->object_depth;

    /* callback */
    if (mod_json_token_invoke_object(tok) != 0) {
      tok->error = mod_json_error_break;
      tok->context = cstr;
      return NULL;
    }

    cstr = mod_json_token_strskp(tok, cstr);
    switch (*cstr) {
      case '}':
        tok->state = mod_json_state_object_finish;
        return (cstr + 1);

      case ']':
        tok->state = mod_json_state_array_finish;
        return (cstr + 1);

      case '\0':
        if (tok->object_depth || tok->array_depth) {
          tok->error = mod_json_error_trunc;
          tok->context = cstr;
        } else {
          tok->state = mod_json_state_finish;
        }
        break;

      case ',':
        if (tok->object_depth || tok->array_depth) {
          mod_json_char_t tag = mod_json_token_tag(tok);

          if (tag == '{') {
            tok->state = mod_json_state_object_half1;
            return (cstr + 1);
          } else if (tag == '[') {
            tok->state = mod_json_state_array_half;
            return (cstr + 1);
          }
        }
        /* FALLTHRU */

      default:
        tok->error = mod_json_error_object;
        tok->context = cstr;
    }
  } else {
    tok->error = mod_json_error_depth;
    tok->context = cstr;
  }
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_null(mod_json_token_t *tok,
                                                    mod_json_cchar_t *cstr) {
  if (!cstr || *cstr == '\0') {
    tok->error = mod_json_error_invalid;
    tok->context = cstr;
    return NULL;
  }

  tok->state = mod_json_state_start;
  return cstr;
}

static inline mod_json_cchar_t *mod_json_token_finish(mod_json_token_t *tok,
                                                      mod_json_cchar_t *cstr) {
  tok->error = mod_json_error_null;
  (void)cstr;
  return NULL;
}

static inline mod_json_cchar_t *mod_json_token_default(mod_json_token_t *tok,
                                                       mod_json_cchar_t *cstr) {
  tok->error = mod_json_error_state;
  tok->context = cstr;
  return NULL;
}

int mod_json_token_parse(mod_json_token_t *tok, mod_json_cchar_t *cstr) {
  while (cstr) {
    switch (tok->state) {
      case mod_json_state_start:
        cstr = mod_json_token_start(tok, cstr);
        break;

      case mod_json_state_array_start:
        cstr = mod_json_token_array_start(tok, cstr);
        break;

      case mod_json_state_array_half:
        cstr = mod_json_token_array_half(tok, cstr);
        break;

      case mod_json_state_array_finish:
        cstr = mod_json_token_array_finish(tok, cstr);
        break;

      case mod_json_state_object_start:
        cstr = mod_json_token_object_start(tok, cstr);
        break;

      case mod_json_state_object_half1:
        cstr = mod_json_token_object_half1(tok, cstr);
        break;

      case mod_json_state_object_half2:
        cstr = mod_json_token_object_half2(tok, cstr);
        break;

      case mod_json_state_object_finish:
        cstr = mod_json_token_object_finish(tok, cstr);
        break;

      case mod_json_state_null:
        cstr = mod_json_token_null(tok, cstr);
        break;

      case mod_json_state_finish:
        cstr = mod_json_token_finish(tok, cstr);
        break;

      default:
        cstr = mod_json_token_default(tok, cstr);
        break;
    }
  }
  return (tok->error == mod_json_error_null ? 0 : -1);
}

static inline int mod_json_parser_insert(mod_json_parser_t *par,
                                         mod_json_size_t depth,
                                         mod_json_value_t *val) {
  if (depth > 0) {
    mod_json_value_t *cur = par->vals[depth - 1];

    switch (cur->type) {
      case mod_json_type_object:
        return (mod_json_object_insert(cur->data.c_obj, par->key, val) ? 0
                                                                       : -1);

      case mod_json_type_array:
        return mod_json_array_push(cur->data.c_arr, val);

      default:
        break;
    }
  }
  return -1;
}

static inline int mod_json_parser_insert_object(mod_json_parser_t *par,
                                                mod_json_size_t depth) {
  mod_json_object_t *obj;
  mod_json_value_t *jval;

  obj = mod_json_object_set_default();
  mod_json_minus_if_false(obj);

  jval = mod_json_value_set_object(obj);
  mod_json_object_unset(obj);
  mod_json_minus_if_false(jval);

  if (depth > 0) {
    int ret = mod_json_parser_insert(par, depth, jval);
    if (ret == 0) {
      par->vals[depth] = jval;
    }
    mod_json_value_unset(jval);
    return ret;
  } else {
    /* It's the root, save the pointer. Don't unset it. */
    par->vals[0] = jval;
  }
  return 0;
}

static inline int mod_json_parser_insert_array(mod_json_parser_t *par,
                                               mod_json_size_t depth) {
  mod_json_array_t *arr;
  mod_json_value_t *jval;

  arr = mod_json_array_set_default();
  mod_json_minus_if_false(arr);

  jval = mod_json_value_set_array(arr);
  mod_json_array_unset(arr);
  mod_json_minus_if_false(jval);

  if (depth > 0) {
    int ret = mod_json_parser_insert(par, depth, jval);
    if (ret == 0) {
      par->vals[depth] = jval;
    }
    mod_json_value_unset(jval);
    return ret;
  } else {
    /* It's the root, save the pointer. Don't unset it. */
    par->vals[0] = jval;
  }
  return 0;
}

static inline void mod_json_token_set_parser(mod_json_token_t *tok,
                                             mod_json_parser_t *par) {
  mod_json_token_set_param(tok, par);
}

static inline mod_json_parser_t *mod_json_token_parser(mod_json_token_t *tok) {
  return (mod_json_parser_t *)mod_json_token_param(tok);
}

static inline int mod_json_parser_event_field(mod_json_token_t *tok,
                                              mod_json_cchar_t *val,
                                              mod_json_size_t len) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  /* unset previous one */
  mod_json_string_unset(parser->key);

  parser->key = mod_json_string_set(val, len);
  return (parser->key ? 0 : -1);
}

static inline int mod_json_parser_event_array(mod_json_token_t *tok) {
  switch (mod_json_token_state(tok)) {
    case mod_json_state_array_finish:
      /* continue */
      return 0;

    case mod_json_state_array_start:
      return mod_json_parser_insert_array(mod_json_token_parser(tok),
                                          mod_json_token_depth(tok));

    default:
      break;
  }
  return -1;
}

static inline int mod_json_parser_event_object(mod_json_token_t *tok) {
  switch (mod_json_token_state(tok)) {
    case mod_json_state_object_finish:
      /* continue */
      return 0;

    case mod_json_state_object_start:
      return mod_json_parser_insert_object(mod_json_token_parser(tok),
                                           mod_json_token_depth(tok));

    default:
      break;
  }
  return -1;
}

static inline int mod_json_parser_event_null(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_null) {
    parser->val_null = mod_json_value_set_null();
    mod_json_minus_if_false(parser->val_null);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_null);
}

static inline int mod_json_parser_event_true(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_true) {
    parser->val_true = mod_json_value_set_boolean(MOD_JSON_TRUE);
    mod_json_minus_if_false(parser->val_true);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_true);
}

static inline int mod_json_parser_event_false(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_false) {
    parser->val_false = mod_json_value_set_boolean(MOD_JSON_FALSE);
    mod_json_minus_if_false(parser->val_false);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_false);
}

static inline int mod_json_parser_event_boolean(mod_json_token_t *tok,
                                                mod_json_boolean_t val) {
  if (!val) {
    return mod_json_parser_event_false(tok);
  }
  return mod_json_parser_event_true(tok);
}

static inline int mod_json_parser_event_zero(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_zero) {
    parser->val_zero = mod_json_value_set_integer(0);
    mod_json_minus_if_false(parser->val_zero);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_zero);
}

static inline int mod_json_parser_event_integer(mod_json_token_t *tok,
                                                mod_json_integer_t val) {
  int ret = -1;

  if (val != 0) {
    mod_json_value_t *jval;

    jval = mod_json_value_set_integer(val);
    if (jval) {
      ret = mod_json_parser_insert(mod_json_token_parser(tok),
                                   mod_json_token_depth(tok), jval);
      mod_json_value_unset(jval);
    }
  } else {
    /* zero event */
    ret = mod_json_parser_event_zero(tok);
  }
  return ret;
}

static inline int mod_json_parser_event_zerof(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_zerof) {
    parser->val_zerof = mod_json_value_set_float(0.0);
    mod_json_minus_if_false(parser->val_zerof);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_zerof);
}

static inline int mod_json_parser_event_float(mod_json_token_t *tok,
                                              mod_json_float_t val) {
  int ret = -1;

  if (val != 0.0) {
    mod_json_value_t *jval;

    jval = mod_json_value_set_float(val);
    if (jval) {
      ret = mod_json_parser_insert(mod_json_token_parser(tok),
                                   mod_json_token_depth(tok), jval);
      mod_json_value_unset(jval);
    }
  } else {
    /* zero event */
    ret = mod_json_parser_event_zerof(tok);
  }
  return ret;
}

static inline int mod_json_parser_event_empty(mod_json_token_t *tok) {
  mod_json_parser_t *parser;

  /* get information */
  parser = mod_json_token_parser(tok);

  if (!parser->val_empty) {
    mod_json_string_t *str;

    str = mod_json_string_set("", 0);
    mod_json_minus_if_false(str);

    parser->val_empty = mod_json_value_set_string(str);
    mod_json_string_unset(str);
    mod_json_minus_if_false(parser->val_empty);
  }
  return mod_json_parser_insert(parser, mod_json_token_depth(tok),
                                parser->val_empty);
}

static inline int mod_json_parser_event_string(mod_json_token_t *tok,
                                               mod_json_cchar_t *val,
                                               mod_json_size_t len) {
  int ret = -1;

  if (len > 0) {
    mod_json_string_t *str;
    mod_json_value_t *jval;

    str = mod_json_string_set(val, len);
    if (str) {
      jval = mod_json_value_set_string(str);
    } else {
      jval = NULL;
    }
    mod_json_string_unset(str);

    if (jval) {
      ret = mod_json_parser_insert(mod_json_token_parser(tok),
                                   mod_json_token_depth(tok), jval);
      mod_json_value_unset(jval);
    }
  } else {
    /* empty event */
    ret = mod_json_parser_event_empty(tok);
  }
  return ret;
}

static int mod_json_parser_event(mod_json_token_t *tok, mod_json_void_t *val,
                                 mod_json_size_t len) {
  switch (tok->event_code) {
    case mod_json_event_field:
      return mod_json_parser_event_field(tok, (mod_json_cchar_t *)val, len);

    case mod_json_event_object:
      return mod_json_parser_event_object(tok);

    case mod_json_event_array:
      return mod_json_parser_event_array(tok);

    case mod_json_event_null:
      return mod_json_parser_event_null(tok);

    case mod_json_event_boolean:
      return mod_json_parser_event_boolean(tok, *(mod_json_boolean_t *)val);

    case mod_json_event_integer:
      return mod_json_parser_event_integer(tok, *(mod_json_integer_t *)val);

    case mod_json_event_float:
      return mod_json_parser_event_float(tok, *(mod_json_float_t *)val);

    case mod_json_event_string:
      return mod_json_parser_event_string(tok, (mod_json_cchar_t *)val, len);

    default:
      break;
  }
  return -1;
}

static inline mod_json_parser_t *mod_json_parser_create(mod_json_size_t depth) {
  mod_json_parser_t *parser;
  mod_json_null_if_false(depth > 0);

  parser = (mod_json_parser_t *)mod_json_malloc(
      depth * sizeof(mod_json_value_t *) + sizeof(mod_json_parser_t));
  mod_json_null_if_false(parser);

  memset(parser, 0, sizeof(mod_json_parser_t));
  parser->vals[0] = NULL;
  return parser;
}

static inline void mod_json_parser_destroy(mod_json_parser_t *par) {
  mod_json_value_unset(par->val_null);
  mod_json_value_unset(par->val_true);
  mod_json_value_unset(par->val_false);
  mod_json_value_unset(par->val_zero);
  mod_json_value_unset(par->val_zerof);
  mod_json_value_unset(par->val_empty);
  mod_json_string_unset(par->key);
  mod_json_free(par);
}

mod_json_value_t *mod_json_parse(mod_json_token_t *tok,
                                 mod_json_cchar_t *cstr) {
  mod_json_parser_t *parser;
  mod_json_value_t *root;
  mod_json_null_if_false(tok && cstr && *cstr);

  parser = mod_json_parser_create(mod_json_token_max_depth(tok));
  mod_json_null_if_false(parser);

  mod_json_token_set_parser(tok, parser);
  mod_json_token_set_event(tok, mod_json_parser_event);

  if (mod_json_token_parse(tok, cstr) == 0) {
    root = parser->vals[0];
  } else {
    /* error occur */
    root = NULL;
    mod_json_value_unset(parser->vals[0]);
  }

  /* clean up */
  mod_json_parser_destroy(parser);

  /* success? */
  return root;
}

mod_json_value_t *mod_json_parse_simply(mod_json_cchar_t *cstr,
                                        mod_json_size_t opts) {
  mod_json_value_t *val;
  mod_json_token_t *tok;
  mod_json_option_t opt;

  opt.options = opts;
  opt.object_depth = 0; /* Use default object depth */
  opt.array_depth = 0;  /* Use default array depth */

  tok = mod_json_token_create(&opt);
  mod_json_null_if_false(tok);

  val = mod_json_parse(tok, cstr);
  mod_json_token_destroy(tok);

  /* value of root */
  return val;
}

static inline int mod_json_dump_null(mod_json_string_t *str) {
  return mod_json_string_add_cstr(str, "null", 4);
}

static inline int mod_json_dump_boolean(mod_json_string_t *str,
                                        mod_json_boolean_t bol) {
  if (!bol) {
    return mod_json_string_add_cstr(str, "false", 5);
  }
  return mod_json_string_add_cstr(str, "true", 4);
}

static inline int mod_json_dump_integer(mod_json_string_t *str,
                                        mod_json_integer_t num) {
  mod_json_char_t buf[32];

  return mod_json_string_add_cstr(str, buf, mod_json_utils_itostr(buf, num));
}

static inline int mod_json_dump_float(mod_json_string_t *str,
                                      mod_json_float_t dbl) {
  mod_json_char_t buf[32];

  return mod_json_string_add_cstr(
      str, buf,
      (mod_json_size_t)mod_json_utils_snprintf(buf, sizeof(buf), "%g", dbl));
}

static inline int mod_json_dump_string(mod_json_string_t *str,
                                       mod_json_string_t *val) {
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '\"'));

  if (val) {
    mod_json_minus_if_ne_zero(mod_json_string_add_jstr(str, val));
  }
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '\"'));
  return 0;
}

static inline int mod_json_dump_value(mod_json_string_t *str,
                                      mod_json_value_t *val);

static inline int mod_json_dump_array(mod_json_string_t *str,
                                      mod_json_array_t *arr) {
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '['));

  if (arr) {
    mod_json_value_t **iter = arr->first;

    for (; iter != arr->last; ++iter) {
      mod_json_minus_if_ne_zero(mod_json_dump_value(str, *iter));
      if (iter + 1 != arr->last) {
        mod_json_minus_if_ne_zero(mod_json_string_add_char(str, ','));
      }
    }
  }
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, ']'));
  return 0;
}

static inline int mod_json_dump_key(mod_json_string_t *str,
                                    mod_json_string_t *key) {
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '\"'));
  mod_json_minus_if_ne_zero(mod_json_string_add_jstr(str, key));
  mod_json_minus_if_ne_zero(mod_json_string_add_cstr(str, "\":", 2));
  return 0;
}

static inline int mod_json_dump_object(mod_json_string_t *str,
                                       mod_json_object_t *obj) {
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '{'));

  if (obj) {
    mod_json_pair_t *iter = obj->first;

    for (; iter != obj->last; ++iter) {
      mod_json_minus_if_ne_zero(mod_json_dump_key(str, iter->key));
      mod_json_minus_if_ne_zero(mod_json_dump_value(str, iter->val));

      if (iter + 1 != obj->last) {
        mod_json_minus_if_ne_zero(mod_json_string_add_char(str, ','));
      }
    }
  }
  mod_json_minus_if_ne_zero(mod_json_string_add_char(str, '}'));
  return 0;
}

static inline int mod_json_dump_value(mod_json_string_t *str,
                                      mod_json_value_t *val) {
  if (val) {
    switch (val->type) {
      case mod_json_type_null:
        return mod_json_dump_null(str);

      case mod_json_type_boolean:
        return mod_json_dump_boolean(str, val->data.c_bool);

      case mod_json_type_integer:
        return mod_json_dump_integer(str, val->data.c_int);

      case mod_json_type_float:
        return mod_json_dump_float(str, val->data.c_float);

      case mod_json_type_string:
        return mod_json_dump_string(str, val->data.c_str);

      case mod_json_type_array:
        return mod_json_dump_array(str, val->data.c_arr);

      case mod_json_type_object:
        return mod_json_dump_object(str, val->data.c_obj);

      default:
        return -1;
    }
  }
  return mod_json_dump_null(str);
}

mod_json_string_t *mod_json_dump(mod_json_value_t *val) {
  mod_json_string_t *str = mod_json_string_set("", 0);
  mod_json_null_if_false(str);

  if (mod_json_unlikely(mod_json_dump_value(str, val) != 0)) {
    /* error occur */
    mod_json_string_unset(str);
    return NULL;
  }
  return str;
}
