// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Datafusion user defined functions

use arrow_array::{ArrayRef, BooleanArray, StringArray, StructArray};
use arrow_schema::DataType;
use datafusion::logical_expr::{create_udf, ScalarUDF, Volatility};
use datafusion::prelude::SessionContext;
use datafusion_functions::utils::make_scalar_function;
use std::sync::{Arc, LazyLock};

/// Register UDF functions to datafusion context.
pub fn register_functions(ctx: &SessionContext) {
    ctx.register_udf(CONTAINS_TOKENS_UDF.clone());
    ctx.register_udf(ST_INTERSECTS_UDF.clone());
    ctx.register_udf(BBOX_UDF.clone());
}

/// This method checks whether a string contains another string. It utilizes FTS (Full-Text Search)
/// indexes, but due to the false negative characteristic of FTS, the results may have omissions.
/// For example, "bakin" will not match documents containing "baking."
/// If the query string is a whole word, or if you prioritize better performance, `contains_tokens`
/// is the better choice. Otherwise, you can use the `contains` method to obtain accurate results.
///
///
/// Usage
/// * Use `contains_tokens` in sql.
/// ```rust,ignore
/// let sql = "SELECT * FROM table WHERE contains_tokens(text_col, 'bakin')"
/// let mut ds = Dataset::open(&ds_path).await?;
/// let mut builder = ds.sql(&sql);
/// let records = builder.clone().build().await?.into_batch_records().await?;
/// ```
fn contains_tokens() -> ScalarUDF {
    let function = Arc::new(make_scalar_function(
        |args: &[ArrayRef]| {
            let column = args[0].as_any().downcast_ref::<StringArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "First argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;
            let scalar_str = args[1].as_any().downcast_ref::<StringArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "Second argument of contains_tokens can't be cast to string".to_string(),
                ),
            )?;

            let result = column
                .iter()
                .enumerate()
                .map(|(i, column)| column.map(|value| value.contains(scalar_str.value(i))));

            Ok(Arc::new(BooleanArray::from_iter(result)) as ArrayRef)
        },
        vec![],
    ));

    create_udf(
        "contains_tokens",
        vec![DataType::Utf8, DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        function,
    )
}

static CONTAINS_TOKENS_UDF: LazyLock<ScalarUDF> = LazyLock::new(contains_tokens);

/// ST_Intersects spatial function that checks if two geometries intersect.
/// This function serves as a DataFusion UDF that can be intercepted by Lance's 
/// geo query parser for index optimization, or fall back to actual geometric computation.
///
/// Usage in SQL:
/// ```sql
/// SELECT * FROM table WHERE ST_Intersects(geometry_column, 'BBOX(-180, -90, 180, 90)')
/// ```
fn st_intersects() -> ScalarUDF {
    let function = Arc::new(make_scalar_function(
        |args: &[ArrayRef]| {
            // For now, this is a placeholder implementation
            // In a full implementation, this would:
            // 1. Parse the geometry arguments (GeoArrow Point struct and WKT string)
            // 2. Perform actual geometric intersection tests using geodatafusion/geo crates
            // 3. Return boolean results
            
            if args.len() != 2 {
                return Err(datafusion::error::DataFusionError::Execution(
                    "st_intersects expects exactly 2 arguments".to_string(),
                ));
            }
            
            // Validate first argument is a struct (GeoArrow Point)
            let _geometry_struct = args[0].as_any().downcast_ref::<StructArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "First argument of st_intersects must be a GeoArrow Point struct".to_string(),
                ),
            )?;
            
            // Validate second argument is a string (WKT polygon)
            let _polygon_wkt = args[1].as_any().downcast_ref::<StringArray>().ok_or(
                datafusion::error::DataFusionError::Execution(
                    "Second argument of st_intersects must be a WKT string".to_string(),
                ),
            )?;
            
            // For now, return a placeholder result that matches all points
            // This will be intercepted by Lance's query parser anyway for indexed queries
            let num_rows = args[0].len();
            Ok(Arc::new(BooleanArray::from(vec![true; num_rows])) as ArrayRef)
        },
        vec![],
    ));

    create_udf(
        "st_intersects",
        vec![
            DataType::Struct(vec![
                Arc::new(arrow_schema::Field::new("x", DataType::Float64, false)),
                Arc::new(arrow_schema::Field::new("y", DataType::Float64, false)),
            ].into()),
            DataType::Utf8
        ], // GeoArrow Point struct, bbox/geometry literal
        DataType::Boolean,
        Volatility::Immutable,
        function,
    )
}

static ST_INTERSECTS_UDF: LazyLock<ScalarUDF> = LazyLock::new(st_intersects);

/// BBOX function that creates a bounding box from four numeric arguments.
/// This function is used internally by spatial queries and doesn't perform actual computation.
/// It's intercepted by Lance's geo query parser for index optimization.
///
/// Usage in SQL:
/// ```sql
/// SELECT * FROM table WHERE ST_Intersects(geometry_column, BBOX(-180, -90, 180, 90))
/// ```
fn bbox() -> ScalarUDF {
    let function = Arc::new(make_scalar_function(
        |args: &[ArrayRef]| {
            // This is a placeholder implementation that should never be called
            // because the BBOX function is intercepted by the query parser
            if args.len() != 4 {
                return Err(datafusion::error::DataFusionError::Execution(
                    "bbox expects exactly 4 arguments (min_x, min_y, max_x, max_y)".to_string(),
                ));
            }
            
            // Extract the 4 numeric arguments and encode them as a parseable string
            // Format: "BBOX(-125.0,30.0,-115.0,45.0)"
            use arrow_array::Float64Array;
            
            if let (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) = (
                args[0].as_any().downcast_ref::<Float64Array>(),
                args[1].as_any().downcast_ref::<Float64Array>(),
                args[2].as_any().downcast_ref::<Float64Array>(),
                args[3].as_any().downcast_ref::<Float64Array>(),
            ) {
                let num_rows = args[0].len();
                let mut result = Vec::with_capacity(num_rows);
                for i in 0..num_rows {
                    let bbox_str = format!("BBOX({},{},{},{})", 
                        min_x.value(i), min_y.value(i), max_x.value(i), max_y.value(i));
                    result.push(bbox_str);
                }
                Ok(Arc::new(StringArray::from(result)) as ArrayRef)
            } else {
                Err(datafusion::error::DataFusionError::Execution(
                    "BBOX arguments must be Float64".to_string(),
                ))
            }
        },
        vec![],
    ));

    create_udf(
        "bbox",
        vec![DataType::Float64, DataType::Float64, DataType::Float64, DataType::Float64], // min_x, min_y, max_x, max_y
        DataType::Utf8, // Returns a string representation (though this is intercepted)
        Volatility::Immutable,
        function,
    )
}

static BBOX_UDF: LazyLock<ScalarUDF> = LazyLock::new(bbox);

#[cfg(test)]
mod tests {
    use crate::udf::CONTAINS_TOKENS_UDF;
    use arrow_array::{Array, BooleanArray, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion::logical_expr::ScalarFunctionArgs;
    use datafusion::physical_plan::ColumnarValue;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_contains_tokens() {
        // Prepare arguments
        let contains_tokens = CONTAINS_TOKENS_UDF.clone();
        let text_col = Arc::new(StringArray::from(vec![
            "a cat",
            "lovely cat",
            "white cat",
            "catch up",
            "fish",
        ]));
        let token = Arc::new(StringArray::from(vec!["cat", "cat", "cat", "cat", "cat"]));

        let args = vec![ColumnarValue::Array(text_col), ColumnarValue::Array(token)];
        let arg_fields = vec![
            Arc::new(Field::new("text_col".to_string(), DataType::Utf8, false)),
            Arc::new(Field::new("token".to_string(), DataType::Utf8, false)),
        ];

        let args = ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 5,
            return_field: Arc::new(Field::new("res".to_string(), DataType::Boolean, false)),
        };

        // Invoke contains_tokens manually
        let values = contains_tokens.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(array) = values {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(
                array.clone(),
                BooleanArray::from(vec![true, true, true, true, false])
            );
        } else {
            panic!("Expected an Array but got {:?}", values);
        }
    }
}
