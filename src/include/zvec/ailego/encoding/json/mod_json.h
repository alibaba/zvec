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

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#if !defined(__cplusplus) && defined(_MSC_VER)
#if !defined(inline)
#define inline __inline
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define MOD_JSON_FALSE (false)
#define MOD_JSON_TRUE (true)
#define MOD_JSON_INFINITY (INFINITY)

/*! JSON Type
 */
enum mod_json_type {
  mod_json_type_null = 0,
  mod_json_type_boolean = 1,
  mod_json_type_integer = 2,
  mod_json_type_float = 3,
  mod_json_type_string = 4,
  mod_json_type_array = 5,
  mod_json_type_object = 6
};

/*! JSON Token State
 */
enum mod_json_state {
  mod_json_state_null = 0,
  mod_json_state_start = 1,
  mod_json_state_finish = 2,
  mod_json_state_array_start = 3,
  mod_json_state_array_half = 4,
  mod_json_state_array_finish = 5,
  mod_json_state_object_start = 6,
  mod_json_state_object_half1 = 7,
  mod_json_state_object_half2 = 8,
  mod_json_state_object_finish = 9,
  mod_json_state_max = 10
};

/*! JSON Token Error Code
 */
enum mod_json_error {
  mod_json_error_null = 0,
  mod_json_error_invalid = 1,
  mod_json_error_state = 2,
  mod_json_error_empty = 3,
  mod_json_error_break = 4,
  mod_json_error_depth = 5,
  mod_json_error_trunc = 6,
  mod_json_error_start = 7,
  mod_json_error_array = 8,
  mod_json_error_object = 9,
  mod_json_error_key = 10,
  mod_json_error_value = 11,
  mod_json_error_quote = 12
};

/*! JSON Token Event
 */
enum mod_json_event {
  mod_json_event_null = 0,
  mod_json_event_field = 1,
  mod_json_event_object = 2,
  mod_json_event_array = 3,
  mod_json_event_boolean = 4,
  mod_json_event_integer = 5,
  mod_json_event_float = 6,
  mod_json_event_string = 7
};

typedef unsigned int mod_json_size_t;
typedef int mod_json_ssize_t;
typedef bool mod_json_boolean_t;
typedef char mod_json_char_t;
typedef const char mod_json_cchar_t;
typedef unsigned char mod_json_uchar_t;
typedef long long mod_json_integer_t;
typedef double mod_json_float_t;
typedef void mod_json_void_t;
typedef enum mod_json_type mod_json_type_t;
typedef union mod_json_any mod_json_any_t;
typedef struct mod_json_value mod_json_value_t;
typedef struct mod_json_string mod_json_string_t;
typedef struct mod_json_array mod_json_array_t;
typedef struct mod_json_object mod_json_object_t;
typedef struct mod_json_pair mod_json_pair_t;
typedef struct mod_json_option mod_json_option_t;
typedef enum mod_json_state mod_json_state_t;
typedef enum mod_json_error mod_json_error_t;
typedef enum mod_json_event mod_json_event_t;
typedef struct mod_json_token mod_json_token_t;

/*! Callback function when parsing JSON
 */
typedef int (*mod_json_event_proc)(mod_json_token_t *tok, mod_json_void_t *val,
                                   mod_json_size_t len);

/*! JSON Any
 */
union mod_json_any {
  mod_json_object_t *c_obj;
  mod_json_array_t *c_arr;
  mod_json_string_t *c_str;
  mod_json_float_t c_float;
  mod_json_boolean_t c_bool;
  mod_json_integer_t c_int;
};

/*! JSON Value
 */
struct mod_json_value {
  mod_json_ssize_t refer;
  mod_json_type_t type;
  mod_json_any_t data;
};

/*! JSON String
 */
struct mod_json_string {
  mod_json_ssize_t refer;
  mod_json_size_t size;
  mod_json_char_t *first;
  mod_json_char_t *last;
};

/*! JSON Array
 */
struct mod_json_array {
  mod_json_ssize_t refer;
  mod_json_size_t size;
  mod_json_value_t **first;
  mod_json_value_t **last;
};

/*! JSON Pair
 */
struct mod_json_pair {
  mod_json_string_t *key;
  mod_json_value_t *val;
};

/*! JSON Object
 */
struct mod_json_object {
  mod_json_ssize_t refer;
  mod_json_size_t size;
  mod_json_pair_t *first;
  mod_json_pair_t *last;
};

#define MOD_JSON_COMMENT 0x0001  /* Enable comments */
#define MOD_JSON_UNSTRICT 0x0002 /* Enable loose JSON string */
#define MOD_JSON_SIMPLE 0x0004   /* Enable simple format */
#define MOD_JSON_SQUOTE 0x0008   /* Enable single quotes support */

/*! JSON Option
 */
struct mod_json_option {
  mod_json_size_t options;
  mod_json_size_t object_depth;
  mod_json_size_t array_depth;
};

/**
 *  \brief           Create and set a JSON null value
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_null(void);

/**
 *  \brief           Create and set a JSON object value
 *  \param obj       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_object(mod_json_object_t *obj);

/**
 *  \brief           Create and set a JSON array value
 *  \param arr       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_array(mod_json_array_t *arr);

/**
 *  \brief           Create and set a JSON string value
 *  \param str       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_string(mod_json_string_t *str);

/**
 *  \brief           Create and set a JSON string buffer
 *  \param buf       The pointer of string buffer
 *  \param len       The length of string buffer
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_buffer(mod_json_cchar_t *buf,
                                            mod_json_size_t len);

/**
 *  \brief           Create and set a JSON integer value
 *  \param num       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_integer(mod_json_integer_t num);

/**
 *  \brief           Create and set a JSON float value
 *  \param dbl       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_float(mod_json_float_t dbl);

/**
 *  \brief           Create and set a JSON boolean value
 *  \param bol       The value to be assigned
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_set_boolean(mod_json_boolean_t bol);

/**
 *  \brief           Assign a JSON value as null
 *  \param val       The pointer of value
 */
void mod_json_value_assign_null(mod_json_value_t *val);

/**
 *  \brief           Assign a JSON value as a object
 *  \param val       The pointer of value
 *  \param obj       The value to be assigned
 */
void mod_json_value_assign_object(mod_json_value_t *val,
                                  mod_json_object_t *obj);

/**
 *  \brief           Assign a JSON value as an array
 *  \param val       The pointer of value
 *  \param arr       The value to be assigned
 */
void mod_json_value_assign_array(mod_json_value_t *val, mod_json_array_t *arr);

/**
 *  \brief           Assign a JSON value as a string
 *  \param val       The pointer of value
 *  \param str       The value to be assigned
 */
void mod_json_value_assign_string(mod_json_value_t *val,
                                  mod_json_string_t *str);

/**
 *  \brief           Assign a JSON value as an integer
 *  \param val       The pointer of value
 *  \param num       The value to be assigned
 */
void mod_json_value_assign_integer(mod_json_value_t *val,
                                   mod_json_integer_t num);

/**
 *  \brief           Assign a JSON value as a float
 *  \param val       The pointer of value
 *  \param dbl       The value to be assigned
 */
void mod_json_value_assign_float(mod_json_value_t *val, mod_json_float_t dbl);

/**
 *  \brief           Assign a JSON value as a boolean
 *  \param val       The pointer of value
 *  \param bol       The value to be assigned
 */
void mod_json_value_assign_boolean(mod_json_value_t *val,
                                   mod_json_boolean_t bol);

/**
 *  \brief           Assign a new JSON value
 *  \param dst       The pointer of destination value (can't be null)
 *  \param src       The pointer of source value (can be null)
 */
void mod_json_value_assign(mod_json_value_t *dst, mod_json_value_t *src);

/**
 *  \brief           Merge a JSON value into another one
 *  \param dst       The pointer of destination value (can't be null)
 *  \param src       The pointer of source value (can be null)
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_value_merge(mod_json_value_t *dst, mod_json_value_t *src);

/**
 *  \brief           Retrieve object of a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates unmatched type or empty.
 */
mod_json_object_t *mod_json_value_object(mod_json_value_t *val);

/**
 *  \brief           Retrieve array of a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates unmatched type or empty.
 */
mod_json_array_t *mod_json_value_array(mod_json_value_t *val);

/**
 *  \brief           Retrieve string of a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates unmatched type or empty.
 */
mod_json_string_t *mod_json_value_string(mod_json_value_t *val);

/**
 *  \brief           Retrieve c-string of a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates unmatched type or empty.
 */
mod_json_cchar_t *mod_json_value_cstring(mod_json_value_t *val);

/**
 *  \brief           Retrieve float of a JSON value
 *  \param val       The pointer of value
 *  \return          It will try converting the unmatched
                     value to float. If nothing be done,
                     returns zero by default.
 */
mod_json_float_t mod_json_value_float(mod_json_value_t *val);

/**
 *  \brief           Retrieve boolean of a JSON value
 *  \param val       The pointer of value
 *  \return          If string, object or array is not empty,
                     number(integer or float) does not equal
                     to zero, it returns true.
 */
mod_json_boolean_t mod_json_value_boolean(mod_json_value_t *val);

/**
 *  \brief           Retrieve integer of a JSON value
 *  \param val       The pointer of value
 *  \return          It will try converting the unmatched
                     value to integer. If nothing be done,
                     returns zero by default.
 */
mod_json_integer_t mod_json_value_integer(mod_json_value_t *val);

/**
 *  \brief           Clone a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_value_clone(mod_json_value_t *val);

/**
 *  \brief           Retrieve non-zero if they are equal
 *  \param lhs       The pointer of left value
 *  \param rhs       The pointer of right value
 *  \return          1 indicates true, 0 indicates false.
 */
mod_json_boolean_t mod_json_value_is_equal(mod_json_value_t *lhs,
                                           mod_json_value_t *rhs);

/**
 *  \brief           Unset or destroy a JSON value
 *  \param val       The pointer of value
 */
void mod_json_value_unset(mod_json_value_t *val);

/**
 *  \brief           Increase reference count of a JSON value
 *  \param val       The pointer of value
 *  \return          The original pointer of value
 */
static inline mod_json_value_t *mod_json_value_get(mod_json_value_t *val) {
  ++val->refer;
  return val;
}

/**
 *  \brief           Decrease reference count of a JSON value
 *  \param val       The pointer of value
 *  \return          The new number of refer-counter
 */
static inline mod_json_ssize_t mod_json_value_put(mod_json_value_t *val) {
  return (--val->refer);
}

/**
 *  \brief           Retrieve refer-counter of a JSON value
 *  \param val       The pointer of value
 *  \return          The number of refer-counter
 */
static inline mod_json_ssize_t mod_json_value_refer(mod_json_value_t *val) {
  return (val ? val->refer : -1);
}

/**
 *  \brief           Set the refer-counter as leaked
 *  \param val       The pointer of value
 */
static inline void mod_json_value_set_leaked(mod_json_value_t *val) {
  val->refer = 0;
}

/**
 *  \brief           Retrieve non-zero if refer-counter is leaked
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_leaked(
    mod_json_value_t *val) {
  return (val->refer <= 0);
}

/**
 *  \brief           Retrieve non-zero if refer-counter is shared
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_shared(
    mod_json_value_t *val) {
  return (val->refer > 1);
}

/**
 *  \brief           Grab (get or clone) a JSON value
 *  \param val       The pointer of value
 *  \return          Null indicates failure
 */
static inline mod_json_value_t *mod_json_value_grab(mod_json_value_t *val) {
  /* Is it leaked? */
  if (!mod_json_value_is_leaked(val)) {
    return mod_json_value_get(val);
  }
  return mod_json_value_clone(val);
}

/**
 *  \brief           Retrieve type of a JSON value
 *  \param val       The pointer of value
 *  \return          The code of type
 */
static inline mod_json_type_t mod_json_value_type(mod_json_value_t *val) {
  return (val->type);
}

/**
 *  \brief           Retrieve non-zero if a JSON value is null
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_null(mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_null : MOD_JSON_TRUE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON array
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_array(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_array : MOD_JSON_FALSE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON object
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_object(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_object : MOD_JSON_FALSE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON string
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_string(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_string : MOD_JSON_FALSE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON float
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_float(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_float : MOD_JSON_FALSE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON boolean
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_boolean(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_boolean : MOD_JSON_FALSE);
}

/**
 *  \brief           Retrieve non-zero if it is a JSON integer
 *  \param val       The pointer of value
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_value_is_integer(
    mod_json_value_t *val) {
  return (val ? val->type == mod_json_type_integer : MOD_JSON_FALSE);
}

/**
 *  \brief           Request a change in capacity
 *  \param str       The pointer of string
 *  \param n         The requested size of capacity
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_string_reserve(mod_json_string_t *str, mod_json_size_t n);

/**
 *  \brief           Create and set a JSON string
 *  \param cstr      The pointer of c-string
 *  \param len       The length of c-string
 *  \return          Null indicates failure.
 */
mod_json_string_t *mod_json_string_set(mod_json_cchar_t *cstr,
                                       mod_json_size_t len);

/**
 *  \brief           Assign new content to a JSON string
 *  \param str       The pointer of string
 *  \param cstr      The pointer of c-string
 *  \param len       The length of c-string
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_string_assign(mod_json_string_t *str, mod_json_cchar_t *cstr,
                           mod_json_size_t len);

/**
 *  \brief           Clone a JSON string
 *  \param str       The pointer of string
 *  \return          Null indicates failure.
 */
static inline mod_json_string_t *mod_json_string_clone(mod_json_string_t *str) {
  return (str ? mod_json_string_set(str->first,
                                    (mod_json_size_t)(str->last - str->first))
              : (mod_json_string_t *)0);
}

/**
 *  \brief           Unset or destroy a JSON string
 *  \param str       The pointer of string
 */
void mod_json_string_unset(mod_json_string_t *str);

/**
 *  \brief           Reset a JSON string
 *  \param str       The pointer of string
 */
void mod_json_string_reset(mod_json_string_t *str);

/**
 *  \brief           Append a c-string to a JSON string
 *  \param str       The pointer of string
 *  \param cstr      The pointer of c-string
 *  \param len       The length of c-string
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_string_append(mod_json_string_t *str, mod_json_cchar_t *cstr,
                           mod_json_size_t len);

/**
 *  \brief           Add a copy of a JSON string
 *  \param str       The main string
 *  \param val       The appended string
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_string_add(mod_json_string_t *str, mod_json_string_t *val);

/**
 *  \brief           Retrieve HASH of a JSON string
 *  \param str       The pointer of string
 *  \return          The value of HASH
 */
mod_json_size_t mod_json_string_hash(mod_json_string_t *str);

/**
 *  \brief           Compare two JSON strings (case sensitive)
 *  \param str1      The first string
 *  \param str2      The second string
 *  \return          0 indicates equal.
 */
int mod_json_string_compare(mod_json_string_t *str1, mod_json_string_t *str2);

/**
 *  \brief           Convert a JSON string to an integer
 *  \param str       The pointer of string
 *  \return          If nothing be done, returns zero by default.
 */
mod_json_integer_t mod_json_string_integer(mod_json_string_t *str);

/**
 *  \brief           Convert a JSON string to a float
 *  \param str       The pointer of string
 *  \return          If nothing be done, returns zero by default.
 */
mod_json_float_t mod_json_string_float(mod_json_string_t *str);

/**
 *  \brief           Encode a JSON string
 *  \param src       The pointer of source string
 *  \return          Null indicates failure.
 */
mod_json_string_t *mod_json_string_encode(mod_json_string_t *src);

/**
 *  \brief           Decode a JSON string
 *  \param src       The pointer of source string
 *  \return          Null indicates failure.
 */
mod_json_string_t *mod_json_string_decode(mod_json_string_t *src);

/**
 *  \brief           Increase reference count of a JSON string
 *  \param str       The pointer of string
 *  \return          The original pointer of string
 */
static inline mod_json_string_t *mod_json_string_get(mod_json_string_t *str) {
  ++str->refer;
  return str;
}

/**
 *  \brief           Decrease reference count of a JSON string
 *  \param str       The pointer of string
 *  \return          The new number of refer-counter
 */
static inline mod_json_ssize_t mod_json_string_put(mod_json_string_t *str) {
  return (--str->refer);
}

/**
 *  \brief           Retrieve refer-counter of a JSON string
 *  \param str       The pointer of string
 *  \return          The number of refer-counter
 */
static inline mod_json_ssize_t mod_json_string_refer(mod_json_string_t *str) {
  return (str ? str->refer : -1);
}

/**
 *  \brief           Set the refer-counter as leaked
 *  \param str       The pointer of string
 */
static inline void mod_json_string_set_leaked(mod_json_string_t *str) {
  str->refer = 0;
}

/**
 *  \brief           Retrieve non-zero if refer-counter is leaked
 *  \param str       The pointer of string
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_string_is_leaked(
    mod_json_string_t *str) {
  return (str->refer <= 0);
}

/**
 *  \brief           Retrieve non-zero if refer-counter is shared
 *  \param str       The pointer of string
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_string_is_shared(
    mod_json_string_t *str) {
  return (str->refer > 1);
}

/**
 *  \brief           Grab (get or clone) a JSON string
 *  \param str       The pointer of string
 *  \return          Null indicates failure
 */
static inline mod_json_string_t *mod_json_string_grab(mod_json_string_t *str) {
  /* Is it leaked? */
  if (!mod_json_string_is_leaked(str)) {
    return mod_json_string_get(str);
  }
  return mod_json_string_clone(str);
}

/**
 *  \brief           Retrieve c-string of a JSON string
 *  \param str       The pointer of string
 *  \return          The pointer of c-string
 */
static inline mod_json_cchar_t *mod_json_string_cstr(mod_json_string_t *str) {
  return (str ? str->first : (mod_json_cchar_t *)0);
}

/**
 *  \brief           Retrieve data pointer of a JSON string
 *  \param str       The pointer of string
 *  \return          The pointer of data
 */
static inline mod_json_char_t *mod_json_string_data(mod_json_string_t *str) {
  return (str ? str->first : (mod_json_char_t *)0);
}

/**
 *  \brief           Retrieve capacity of a JSON string
 *  \param str       The pointer of string
 *  \return          The size of allocated storage
 */
static inline mod_json_size_t mod_json_string_capacity(mod_json_string_t *str) {
  return (str ? (str->size - 1) : 0);
}

/**
 *  \brief           Retrieve length of a JSON string
 *  \param str       The pointer of string
 *  \return          The length of string
 */
static inline mod_json_size_t mod_json_string_length(mod_json_string_t *str) {
  return (str ? (mod_json_size_t)(str->last - str->first) : 0);
}

/**
 *  \brief           Retrieve non-zero if a JSON string is empty
 *  \param str       The pointer of string
 *  \return          0 indicates non-empty
 */
static inline mod_json_boolean_t mod_json_string_empty(mod_json_string_t *str) {
  return (mod_json_string_length(str) == 0);
}

/**
 *  \brief           Create and set a JSON array
 *  \param size      The initialized size of array
 *  \return          Null indicates failure.
 */
mod_json_array_t *mod_json_array_set(mod_json_size_t size);

/**
 *  \brief           Clone a JSON array
 *  \param arr       The pointer of array
 *  \return          Null indicates failure.
 */
mod_json_array_t *mod_json_array_clone(mod_json_array_t *arr);

/**
 *  \brief           Retrieve non-zero if they are equal
 *  \param lhs       The pointer of left array
 *  \param rhs       The pointer of right array
 *  \return          1 indicates true, 0 indicates false.
 */
mod_json_boolean_t mod_json_array_is_equal(mod_json_array_t *lhs,
                                           mod_json_array_t *rhs);

/**
 *  \brief           Unset or destroy a JSON array
 *  \param arr       The pointer of array
 */
void mod_json_array_unset(mod_json_array_t *arr);

/**
 *  \brief           Reset a JSON array
 *  \param arr       The pointer of array
 */
void mod_json_array_reset(mod_json_array_t *arr);

/**
 *  \brief           Create and set a JSON array (default parameters)
 *  \param size      The initialized size of array
 *  \return          Null indicates failure.
 */
static inline mod_json_array_t *mod_json_array_set_default(void) {
  return mod_json_array_set(0);
}

/**
 *  \brief           Increase reference count of a JSON array
 *  \param arr       The pointer of array
 *  \return          The original pointer of array
 */
static inline mod_json_array_t *mod_json_array_get(mod_json_array_t *arr) {
  ++arr->refer;
  return arr;
}

/**
 *  \brief           Decrease reference count of a JSON array
 *  \param str       The pointer of array
 *  \return          The new number of refer-counter
 */
static inline mod_json_ssize_t mod_json_array_put(mod_json_array_t *arr) {
  return (--arr->refer);
}

/**
 *  \brief           Retrieve refer-counter of a JSON array
 *  \param arr       The pointer of array
 *  \return          The number of refer-counter
 */
static inline mod_json_ssize_t mod_json_array_refer(mod_json_array_t *arr) {
  return (arr ? arr->refer : -1);
}

/**
 *  \brief           Set the refer-counter as leaked
 *  \param arr       The pointer of array
 */
static inline void mod_json_array_set_leaked(mod_json_array_t *arr) {
  arr->refer = 0;
}

/**
 *  \brief           Retrieve non-zero if refer-counter is leaked
 *  \param arr       The pointer of array
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_array_is_leaked(
    mod_json_array_t *arr) {
  return (arr->refer <= 0);
}

/**
 *  \brief           Retrieve non-zero if refer-counter is shared
 *  \param arr       The pointer of array
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_array_is_shared(
    mod_json_array_t *arr) {
  return (arr->refer > 1);
}

/**
 *  \brief           Grab (get or clone) a JSON array
 *  \param arr       The pointer of array
 *  \return          Null indicates failure
 */
static inline mod_json_array_t *mod_json_array_grab(mod_json_array_t *arr) {
  /* Is it leaked? */
  if (!mod_json_array_is_leaked(arr)) {
    return mod_json_array_get(arr);
  }
  return mod_json_array_clone(arr);
}

/**
 *  \brief           Retrieve count of elements in a JSON array
 *  \param arr       The pointer of array
 *  \return          The count of elements
 */
static inline mod_json_size_t mod_json_array_count(mod_json_array_t *arr) {
  return (arr ? (mod_json_size_t)(arr->last - arr->first) : 0);
}

/**
 *  \brief           Retrieve capacity of a JSON array
 *  \param arr       The pointer of array
 *  \return          The size of allocated storage
 */
static inline mod_json_size_t mod_json_array_capacity(mod_json_array_t *arr) {
  return (arr ? arr->size : 0);
}

/**
 *  \brief           Retrieve non-zero if a JSON array is empty
 *  \param arr       The pointer of array
 *  \return          0 indicates non-empty
 */
static inline mod_json_boolean_t mod_json_array_empty(mod_json_array_t *arr) {
  return (mod_json_array_count(arr) == 0);
}

/**
 *  \brief           Retrieve the begin of a JSON array
 *  \param arr       The pointer of array
 *  \return          The pointer of begin
 */
static inline mod_json_value_t **mod_json_array_begin(mod_json_array_t *arr) {
  return (arr->first);
}

/**
 *  \brief           Retrieve the reverse begin of a JSON array
 *  \param arr       The pointer of array
 *  \return          The pointer of reverse begin
 */
static inline mod_json_value_t **mod_json_array_rbegin(mod_json_array_t *arr) {
  return (arr->last - 1);
}

/**
 *  \brief           Retrieve the end of a JSON array
 *  \param arr       The pointer of array
 *  \return          The pointer of end
 */
static inline mod_json_value_t **mod_json_array_end(mod_json_array_t *arr) {
  return (arr->last);
}

/**
 *  \brief           Retrieve the reverse end of a JSON array
 *  \param arr       The pointer of array
 *  \return          The pointer of reverse end
 */
static inline mod_json_value_t **mod_json_array_rend(mod_json_array_t *arr) {
  return (arr->first - 1);
}

/**
 *  \brief           Request a change in capacity
 *  \param arr       The pointer of array
 *  \param n         The requested size of capacity
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_array_reserve(mod_json_array_t *arr, mod_json_size_t n);

/**
 *  \brief           Reverse the order of the elements in an array
 *  \param arr       The pointer of array
 *  \return          0 indicates success, -1 indicates failure.
 */
void mod_json_array_reverse(mod_json_array_t *arr);

/**
 *  \brief           Push a value into a JSON array
 *  \param arr       The pointer of array
 *  \param val       The pointer of value
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_array_push(mod_json_array_t *arr, mod_json_value_t *val);

/**
 *  \brief           Pop the last element from a JSON array
 *  \param arr       The pointer of array
 */
void mod_json_array_pop(mod_json_array_t *arr);

/**
 *  \brief           Remove the first element of a JSON array
 *  \param arr       The pointer of array
 */
void mod_json_array_shift(mod_json_array_t *arr);

/**
 *  \brief           Retrieve a value in JSON array
 *  \param arr       The pointer of array
 *  \param id        The index (start from zero)
 *  \return          Null indicates no one be found.
 */
mod_json_value_t *mod_json_array_at(mod_json_array_t *arr, mod_json_size_t id);

/**
 *  \brief           Merge a JSON array into another one
 *  \param dst       The pointer of destination array (can't be null)
 *  \param src       The pointer of source array (can't be null)
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_array_merge(mod_json_array_t *dst, mod_json_array_t *src);

/**
 *  \brief           Resize a JSON array so that it contains n elements
 *  \param arr       The pointer of array
 *  \param n         The new size, expressed in number of elements
 *  \param val       The pointer of value assigned (can be null)
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_array_resize(mod_json_array_t *arr, mod_json_size_t n,
                          mod_json_value_t *val);

/**
 *  \brief           Retrieve key of a JSON pair
 *  \param pair      The pointer of pair
 *  \return          The key of pair
 */
static inline mod_json_string_t *mod_json_pair_key(mod_json_pair_t *pair) {
  return (pair->key);
}

/**
 *  \brief           Retrieve value of a JSON pair
 *  \param pair      The pointer of pair
 *  \return          The value of pair
 */
static inline mod_json_value_t *mod_json_pair_value(mod_json_pair_t *pair) {
  return (pair->val);
}

/**
 *  \brief           Create and set a JSON object
 *  \param size      The initialized size of object
 *  \return          Null indicates failure.
 */
mod_json_object_t *mod_json_object_set(mod_json_size_t size);

/**
 *  \brief           Clone a JSON object
 *  \param obj       The pointer of object
 *  \return          Null indicates failure.
 */
mod_json_object_t *mod_json_object_clone(mod_json_object_t *obj);

/**
 *  \brief           Retrieve non-zero if they are equal
 *  \param lhs       The pointer of left object
 *  \param rhs       The pointer of right object
 *  \return          1 indicates true, 0 indicates false.
 */
mod_json_boolean_t mod_json_object_is_equal(mod_json_object_t *lhs,
                                            mod_json_object_t *rhs);

/**
 *  \brief           Unset or destroy a JSON object
 *  \param obj       The pointer of object
 */
void mod_json_object_unset(mod_json_object_t *obj);

/**
 *  \brief           Reset a JSON object
 *  \param obj       The pointer of object
 */
void mod_json_object_reset(mod_json_object_t *obj);

/**
 *  \brief           Create and set a JSON object (default parameters)
 *  \return          Null indicates failure.
 */
static inline mod_json_object_t *mod_json_object_set_default(void) {
  return mod_json_object_set(0);
}

/**
 *  \brief           Increase reference count of a JSON object
 *  \param obj       The pointer of object
 *  \return          The original pointer of object
 */
static inline mod_json_object_t *mod_json_object_get(mod_json_object_t *obj) {
  ++obj->refer;
  return obj;
}

/**
 *  \brief           Decrease reference count of a JSON object
 *  \param str       The pointer of object
 *  \return          The new number of refer-counter
 */
static inline mod_json_ssize_t mod_json_object_put(mod_json_object_t *obj) {
  return (--obj->refer);
}

/**
 *  \brief           Retrieve refer-counter of a JSON object
 *  \param obj       The pointer of object
 *  \return          The number of refer-counter
 */
static inline mod_json_ssize_t mod_json_object_refer(mod_json_object_t *obj) {
  return (obj ? obj->refer : -1);
}

/**
 *  \brief           Set the refer-counter as leaked
 *  \param obj       The pointer of object
 */
static inline void mod_json_object_set_leaked(mod_json_object_t *obj) {
  obj->refer = 0;
}

/**
 *  \brief           Retrieve non-zero if refer-counter is leaked
 *  \param obj       The pointer of object
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_object_is_leaked(
    mod_json_object_t *obj) {
  return (obj->refer <= 0);
}

/**
 *  \brief           Retrieve non-zero if refer-counter is shared
 *  \param obj       The pointer of object
 *  \return          1 indicates TRUE, 0 indicates FALSE
 */
static inline mod_json_boolean_t mod_json_object_is_shared(
    mod_json_object_t *obj) {
  return (obj->refer > 1);
}

/**
 *  \brief           Grab (get or clone) a JSON object
 *  \param obj       The pointer of object
 *  \return          Null indicates failure
 */
static inline mod_json_object_t *mod_json_object_grab(mod_json_object_t *obj) {
  /* Is it leaked? */
  if (!mod_json_object_is_leaked(obj)) {
    return mod_json_object_get(obj);
  }
  return mod_json_object_clone(obj);
}

/**
 *  \brief           Retrieve count of elements in a JSON object
 *  \param obj       The pointer of object
 *  \return          The count of elements
 */
static inline mod_json_size_t mod_json_object_count(mod_json_object_t *obj) {
  return (obj ? (mod_json_size_t)(obj->last - obj->first) : 0);
}

/**
 *  \brief           Retrieve non-zero if a JSON object is empty
 *  \param obj       The pointer of object
 *  \return          0 indicates non-empty
 */
static inline mod_json_boolean_t mod_json_object_empty(mod_json_object_t *obj) {
  return (mod_json_object_count(obj) == 0);
}

/**
 *  \brief           Retrieve the begin of a JSON object
 *  \param obj       The pointer of object
 *  \return          The pointer of begin
 */
static inline mod_json_pair_t *mod_json_object_begin(mod_json_object_t *obj) {
  return (obj->first);
}

/**
 *  \brief           Retrieve the reverse begin of a JSON object
 *  \param obj       The pointer of object
 *  \return          The pointer of reverse begin
 */
static inline mod_json_pair_t *mod_json_object_rbegin(mod_json_object_t *obj) {
  return (obj->last - 1);
}

/**
 *  \brief           Retrieve the end of a JSON object
 *  \param obj       The pointer of object
 *  \return          The pointer of end
 */
static inline mod_json_pair_t *mod_json_object_end(mod_json_object_t *obj) {
  return (obj->last);
}

/**
 *  \brief           Retrieve the reverse end of a JSON object
 *  \param obj       The pointer of object
 *  \return          The pointer of reverse end
 */
static inline mod_json_pair_t *mod_json_object_rend(mod_json_object_t *obj) {
  return (obj->first - 1);
}

/**
 *  \brief           Insert a pair into a JSON object
 *  \param obj       The pointer of object
 *  \param key       The string of key
 *  \param val       The pointer of value
 *  \return          The pair inserted, Null indicates failure.
 */
mod_json_pair_t *mod_json_object_insert(mod_json_object_t *obj,
                                        mod_json_string_t *key,
                                        mod_json_value_t *val);

/**
 *  \brief           Assign a pair into a JSON object
 *  \param obj       The pointer of object
 *  \param key       The string of key
 *  \param val       The pointer of value
 *  \return          The pair assigned, Null indicates failure.
 */
mod_json_pair_t *mod_json_object_assign(mod_json_object_t *obj,
                                        mod_json_string_t *key,
                                        mod_json_value_t *val);

/**
 *  \brief           Touch a pair in a JSON object
 *  \param obj       The pointer of object
 *  \param key       The c-string of key
 */
mod_json_pair_t *mod_json_object_touch(mod_json_object_t *obj,
                                       mod_json_cchar_t *key);

/**
 *  \brief           Erase a pair from a JSON object
 *  \param obj       The pointer of object
 *  \param key       The c-string of key
 */
void mod_json_object_erase(mod_json_object_t *obj, mod_json_cchar_t *key);

/**
 *  \brief           Get a value in a JSON object
 *  \param obj       The pointer of object
 *  \param key       The c-string of key
 *  \return          Null indicates failure.
 */
mod_json_value_t *mod_json_object_at(mod_json_object_t *obj,
                                     mod_json_cchar_t *key);

/**
 *  \brief           Find a pair in a JSON object
 *  \param obj       The pointer of object
 *  \param key       The c-string of key
 *  \return          Null indicates failure.
 */
mod_json_pair_t *mod_json_object_find(mod_json_object_t *obj,
                                      mod_json_cchar_t *key);

/**
 *  \brief           Merge a JSON object into another one
 *  \param dst       The pointer of destination object (can't be null)
 *  \param src       The pointer of source object (can't be null)
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_object_merge(mod_json_object_t *dst, mod_json_object_t *src);

/**
 *  \brief           Create a JSON token
 *  \param opt       The options of parser
 *  \return          The pointer of token, Null indicates failure.
 */
mod_json_token_t *mod_json_token_create(mod_json_option_t *opt);

/**
 *  \brief           Destroy a JSON token
 *  \param tok       The pointer of token
 */
void mod_json_token_destroy(mod_json_token_t *tok);

/**
 *  \brief           Parse a c-string with a JSON token
 *  \param tok       The pointer of token
 *  \param cstr      The pointer of c-string
 *  \return          0 indicates success, -1 indicates failure.
 */
int mod_json_token_parse(mod_json_token_t *tok, mod_json_cchar_t *cstr);

/**
 *  \brief           Retrieve error of a JSON token
 *  \param tok       The pointer of token
 *  \return          The code of error
 */
mod_json_error_t mod_json_token_error(mod_json_token_t *tok);

/**
 *  \brief           Retrieve error context of a JSON token
 *  \param tok       The pointer of token
 *  \return          The pointer of context, null indicates non-errors
 */
mod_json_cchar_t *mod_json_token_context(mod_json_token_t *tok);

/**
 *  \brief           Retrieve state of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of state
 */
mod_json_state_t mod_json_token_state(mod_json_token_t *tok);

/**
 *  \brief           Retrieve object depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of object depth
 */
mod_json_size_t mod_json_token_object_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve array depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of array depth
 */
mod_json_size_t mod_json_token_array_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve max object depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of max object depth
 */
mod_json_size_t mod_json_token_max_object_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve max array depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of max array depth
 */
mod_json_size_t mod_json_token_max_array_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of depth
 */
mod_json_size_t mod_json_token_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve max depth of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of max depth
 */
mod_json_size_t mod_json_token_max_depth(mod_json_token_t *tok);

/**
 *  \brief           Retrieve parameter of a JSON token
 *  \param tok       The pointer of token
 *  \return          The value of parameter
 */
mod_json_void_t *mod_json_token_param(mod_json_token_t *tok);

/**
 *  \brief           Set parameter of a JSON token
 *  \param tok       The pointer of token
 *  \param param     The value of parameter
 */
void mod_json_token_set_param(mod_json_token_t *tok, mod_json_void_t *param);

/**
 *  \brief           Register callback function of a JSON token
 *  \param tok       The pointer of token
 *  \param proc      The pointer of callback function
 */
void mod_json_token_set_event(mod_json_token_t *tok, mod_json_event_proc proc);

/**
 *  \brief           Retrieve event code of a JSON token
 *  \param tok       The pointer of token
 *  \return          The code of event
 */
mod_json_event_t mod_json_token_event(mod_json_token_t *tok);

/**
 *  \brief           Parse a c-string with a JSON token
 *  \param tok       The pointer of token
 *  \param cstr      The pointer of c-string
 *  \return          The pointer of value, Null indicates failure.
 */
mod_json_value_t *mod_json_parse(mod_json_token_t *tok, mod_json_cchar_t *cstr);

/**
 *  \brief           Parse a c-string simply
 *  \param cstr      The pointer of c-string
 *  \param opts      The options of parser
 *  \return          The pointer of value, Null indicates failure.
 */
mod_json_value_t *mod_json_parse_simply(mod_json_cchar_t *cstr,
                                        mod_json_size_t opts);

/**
 *  \brief           Dump a JSON value in string
 *  \param val       The pointer of value
 *  \return          Null indicates failure.
 */
mod_json_string_t *mod_json_dump(mod_json_value_t *val);

#if defined(__cplusplus)
} /* extern "C" */
#endif
