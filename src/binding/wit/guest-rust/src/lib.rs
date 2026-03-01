// Copyright 2025-present the zvec project
//
// Rust guest implementation of the zvec WIT interface.
// Stub implementation demonstrating the component shape.

#[allow(warnings)]
mod bindings {
    wit_bindgen::generate!({
        path: "../wit",
        world: "zvec-guest",
    });
}

use bindings::exports::zvec::db::types::*;

// --- Schema ---
pub struct MySchema {
    name: String,
}

impl GuestSchema for MySchema {
    fn new(name: String) -> Self {
        MySchema { name }
    }

    fn add_field(&self, _name: String, _data_type: DataType, _dimension: u32) -> Result<(), ZvecError> {
        Ok(())
    }
}

// --- Doc ---
pub struct MyDoc {
    pk: String,
    score: f32,
}

impl GuestDoc for MyDoc {
    fn new() -> Self {
        MyDoc {
            pk: String::new(),
            score: 0.0,
        }
    }

    fn set_pk(&self, _pk: String) {}

    fn pk(&self) -> String {
        self.pk.clone()
    }

    fn set_score(&self, _score: f32) {}

    fn score(&self) -> f32 {
        self.score
    }

    fn set_string(&self, _field: String, _value: String) -> Result<(), ZvecError> {
        Ok(())
    }

    fn set_int32(&self, _field: String, _value: i32) -> Result<(), ZvecError> {
        Ok(())
    }

    fn set_float(&self, _field: String, _value: f32) -> Result<(), ZvecError> {
        Ok(())
    }

    fn set_vector(&self, _field: String, _data: Vec<f32>) -> Result<(), ZvecError> {
        Ok(())
    }
}

// --- Collection ---
pub struct MyCollection {
    path: String,
}

impl GuestCollection for MyCollection {
    fn create(path: String, _schema: SchemaBorrow<'_>) -> Result<Collection, ZvecError> {
        Ok(Collection::new(MyCollection { path }))
    }

    fn open(path: String) -> Result<Collection, ZvecError> {
        Ok(Collection::new(MyCollection { path }))
    }

    fn upsert(&self, _docs: Vec<DocBorrow<'_>>) -> Result<(), ZvecError> {
        Ok(())
    }

    fn insert(&self, _docs: Vec<DocBorrow<'_>>) -> Result<(), ZvecError> {
        Ok(())
    }

    fn update(&self, _docs: Vec<DocBorrow<'_>>) -> Result<(), ZvecError> {
        Ok(())
    }

    fn delete(&self, _pks: Vec<String>) -> Result<(), ZvecError> {
        Ok(())
    }

    fn fetch(&self, _pks: Vec<String>) -> Result<Vec<QueryResult>, ZvecError> {
        Ok(Vec::new())
    }

    fn query(
        &self,
        _field: String,
        _vector: Vec<f32>,
        _topk: i32,
    ) -> Result<Vec<QueryResult>, ZvecError> {
        Ok(Vec::new())
    }

    fn flush(&self) -> Result<(), ZvecError> {
        Ok(())
    }

    fn stats(&self) -> Result<u64, ZvecError> {
        Ok(0)
    }

    fn destroy_physical(&self) -> Result<(), ZvecError> {
        Ok(())
    }
}

bindings::export!(MyComponent with_types_in bindings);

pub struct MyComponent;

impl bindings::exports::zvec::db::types::Guest for MyComponent {
    type Schema = MySchema;
    type Doc = MyDoc;
    type Collection = MyCollection;
}
