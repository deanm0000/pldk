use std::cmp::Ordering;
use std::collections::HashMap;

use polars::prelude::*;

/// DuckLake bounds are conservative, not necessarily exact. `value_count`
/// counts non-null values, separately from `null_count`.
#[derive(Clone, Debug)]
pub struct ColumnStats {
    pub dtype: DataType,
    pub min: Option<String>,
    pub max: Option<String>,
    pub null_count: Option<i64>,
    pub value_count: Option<i64>,
    pub contains_nan: Option<bool>,
}

/// Returns false only when the statistics rule out every true result.
pub fn may_match(predicate: &Expr, stats: &HashMap<String, ColumnStats>) -> bool {
    match predicate {
        Expr::BinaryExpr { left, op, right } => match op {
            Operator::And | Operator::LogicalAnd => {
                may_match(left, stats) && may_match(right, stats)
            }
            Operator::Or | Operator::LogicalOr => may_match(left, stats) || may_match(right, stats),
            Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq => {
                let (name, literal, op) = match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(name), Expr::Literal(literal)) => (name, literal, *op),
                    (Expr::Literal(literal), Expr::Column(name)) => (name, literal, reverse(*op)),
                    _ => return true,
                };
                let Some(column) = stats.get(name.as_str()) else {
                    return true;
                };
                compare(column, literal, op)
            }
            _ => true,
        },
        Expr::Function {
            input,
            function: FunctionExpr::Boolean(function),
        } => {
            if let BooleanFunction::IsBetween { closed } = function {
                if input.len() != 3 {
                    return true;
                }
                let lower = match closed {
                    ClosedInterval::Both | ClosedInterval::Left => Operator::GtEq,
                    _ => Operator::Gt,
                };
                let upper = match closed {
                    ClosedInterval::Both | ClosedInterval::Right => Operator::LtEq,
                    _ => Operator::Lt,
                };
                return may_match(
                    &Expr::BinaryExpr {
                        left: Arc::new(input[0].clone()),
                        op: lower,
                        right: Arc::new(input[1].clone()),
                    },
                    stats,
                ) && may_match(
                    &Expr::BinaryExpr {
                        left: Arc::new(input[0].clone()),
                        op: upper,
                        right: Arc::new(input[2].clone()),
                    },
                    stats,
                );
            }
            if let BooleanFunction::IsIn { nulls_equal } = function {
                if input.len() != 2 {
                    return true;
                }
                let (Expr::Column(name), Expr::Literal(literal)) = (&input[0], &input[1]) else {
                    return true;
                };
                let Some(column) = stats.get(name.as_str()) else {
                    return true;
                };
                let values = match literal {
                    LiteralValue::Series(series) => series.clone().into_inner(),
                    _ => match literal.to_any_value() {
                        Some(AnyValue::List(series)) => series,
                        _ => return true,
                    },
                };
                return (0..values.len()).any(|i| {
                    let Ok(value) = values.get(i) else {
                        return true;
                    };
                    if value.is_null() {
                        return *nulls_equal && column.null_count != Some(0);
                    }
                    let literal = LiteralValue::Scalar(Scalar::new(
                        values.dtype().clone(),
                        value.into_static(),
                    ));
                    compare(column, &literal, Operator::Eq)
                });
            }
            if input.len() != 1 {
                return true;
            }
            let Expr::Column(name) = &input[0] else {
                return true;
            };
            let Some(column) = stats.get(name.as_str()) else {
                return true;
            };
            if !scalar_dtype(&column.dtype) || !valid_counts(column) {
                return true;
            }
            match function {
                BooleanFunction::IsNull => column.null_count != Some(0),
                BooleanFunction::IsNotNull => column.value_count != Some(0),
                _ => true,
            }
        }
        _ => true,
    }
}

fn reverse(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

fn scalar_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int128
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::String
            | DataType::Date
            | DataType::Time
            | DataType::Datetime(..)
            | DataType::Decimal(..)
    )
}

fn valid_counts(stats: &ColumnStats) -> bool {
    stats.null_count.is_none_or(|n| n >= 0) && stats.value_count.is_none_or(|n| n >= 0)
}

fn integer_in_dtype(value: i128, dtype: &DataType) -> bool {
    match dtype {
        DataType::Int8 => i8::try_from(value).is_ok(),
        DataType::Int16 => i16::try_from(value).is_ok(),
        DataType::Int32 => i32::try_from(value).is_ok(),
        DataType::Int64 => i64::try_from(value).is_ok(),
        DataType::Int128 => true,
        DataType::UInt8 => u8::try_from(value).is_ok(),
        DataType::UInt16 => u16::try_from(value).is_ok(),
        DataType::UInt32 => u32::try_from(value).is_ok(),
        DataType::UInt64 => u64::try_from(value).is_ok(),
        _ => false,
    }
}

fn integer_literal(literal: &LiteralValue, dtype: &DataType) -> Option<i128> {
    let value = match literal {
        LiteralValue::Dyn(_) => match literal.get_datatype() {
            DataType::Unknown(UnknownKind::Int(value)) => value,
            _ => return None,
        },
        LiteralValue::Scalar(scalar) if scalar.dtype() == dtype => match scalar.as_any_value() {
            AnyValue::Int8(v) => v.into(),
            AnyValue::Int16(v) => v.into(),
            AnyValue::Int32(v) => v.into(),
            AnyValue::Int64(v) => v.into(),
            AnyValue::Int128(v) => v,
            AnyValue::UInt8(v) => v.into(),
            AnyValue::UInt16(v) => v.into(),
            AnyValue::UInt32(v) => v.into(),
            AnyValue::UInt64(v) => v.into(),
            _ => return None,
        },
        _ => return None,
    };
    integer_in_dtype(value, dtype).then_some(value)
}

// A lower and upper bound can prove a singleton even when neither bound is exact:
// every actual value must lie between them.
fn range_may_match<T: PartialOrd>(min: T, max: T, value: T, op: Operator) -> bool {
    if min
        .partial_cmp(&max)
        .is_none_or(|ord| ord == Ordering::Greater)
    {
        return true;
    }
    match op {
        Operator::Eq => !(value < min || value > max),
        Operator::NotEq => !(min == value && max == value),
        Operator::Lt => min < value,
        Operator::LtEq => min <= value,
        Operator::Gt => max > value,
        Operator::GtEq => max >= value,
        _ => true,
    }
}

fn compare(stats: &ColumnStats, literal: &LiteralValue, op: Operator) -> bool {
    if !scalar_dtype(&stats.dtype) || !valid_counts(stats) {
        return true;
    }
    // Ordinary comparisons with null never produce true, irrespective of bounds.
    if matches!(literal, LiteralValue::Scalar(scalar) if scalar.is_null()) {
        return false;
    }
    if stats.value_count == Some(0) {
        return false;
    }
    let (Some(min), Some(max)) = (&stats.min, &stats.max) else {
        return true;
    };

    match &stats.dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => {
            let (Ok(min), Ok(max), Some(value)) = (
                min.parse::<i128>(),
                max.parse::<i128>(),
                integer_literal(literal, &stats.dtype),
            ) else {
                return true;
            };
            if !integer_in_dtype(min, &stats.dtype) || !integer_in_dtype(max, &stats.dtype) {
                return true;
            }
            range_may_match(min, max, value, op)
        }
        DataType::Float32 | DataType::Float64 => {
            let bounds = match stats.dtype {
                DataType::Float32 => min
                    .parse::<f32>()
                    .ok()
                    .zip(max.parse::<f32>().ok())
                    .map(|(a, b)| (f64::from(a), f64::from(b))),
                _ => min.parse::<f64>().ok().zip(max.parse::<f64>().ok()),
            };
            let value = match literal {
                LiteralValue::Dyn(_) if stats.dtype == DataType::Float64 => {
                    match literal.get_datatype() {
                        DataType::Unknown(UnknownKind::Float) => match literal.to_any_value() {
                            Some(AnyValue::Float64(v)) => Some(v),
                            _ => None,
                        },
                        _ => None,
                    }
                }
                LiteralValue::Scalar(scalar) if scalar.dtype() == &stats.dtype => {
                    match scalar.as_any_value() {
                        AnyValue::Float32(v) => Some(f64::from(v)),
                        AnyValue::Float64(v) => Some(v),
                        _ => None,
                    }
                }
                _ => None,
            };
            match (bounds, value) {
                (Some((min, max)), Some(value)) if !min.is_nan() && !max.is_nan() => {
                    // NaNs are above finite values in Polars' total ordering.
                    if value.is_nan() {
                        return match op {
                            Operator::Eq | Operator::GtEq => stats.contains_nan != Some(false),
                            Operator::Gt => false,
                            _ => true,
                        };
                    }
                    if stats.contains_nan != Some(false)
                        && matches!(op, Operator::Gt | Operator::GtEq | Operator::NotEq)
                    {
                        return true;
                    }
                    range_may_match(min, max, value, op)
                }
                _ => true,
            }
        }
        DataType::String => match literal.extract_str() {
            Some(value) => range_may_match(min.as_str(), max.as_str(), value, op),
            None => true,
        },
        DataType::Boolean => {
            let parse = |s: &str| match s {
                "0" | "false" => Some(false),
                "1" | "true" => Some(true),
                _ => None,
            };
            let (Some(min), Some(max), Some(value)) = (parse(min), parse(max), literal.bool())
            else {
                return true;
            };
            range_may_match(min, max, value, op)
        }
        DataType::Date | DataType::Time | DataType::Datetime(..) | DataType::Decimal(..) => {
            let LiteralValue::Scalar(scalar) = literal else {
                return true;
            };
            if scalar.dtype() != &stats.dtype {
                return true;
            }
            let column = crate::types::Column {
                id: 0,
                name: String::new(),
                dtype: stats.dtype.clone(),
                initial_default: None,
                bytea_text: false,
            };
            let Ok(bounds) =
                crate::types::text_series(&column, &[Some(min.clone()), Some(max.clone())])
            else {
                return true;
            };
            let (Ok(min), Ok(max)) = (bounds.get(0), bounds.get(1)) else {
                return true;
            };
            range_may_match(min, max, scalar.as_any_value(), op)
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(dtype: DataType, min: &str, max: &str) -> HashMap<String, ColumnStats> {
        HashMap::from([(
            "x".into(),
            ColumnStats {
                dtype,
                min: Some(min.into()),
                max: Some(max.into()),
                null_count: Some(0),
                value_count: Some(5),
                contains_nan: Some(false),
            },
        )])
    }

    #[test]
    fn comparisons_and_operand_order() {
        let s = stats(DataType::Int64, "10", "20");
        assert!(!may_match(&col("x").eq(lit(9i64)), &s));
        assert!(may_match(&col("x").eq(lit(10i64)), &s));
        assert!(!may_match(&col("x").lt(lit(10i64)), &s));
        assert!(may_match(&col("x").lt_eq(lit(10i64)), &s));
        assert!(!may_match(&lit(20i64).lt(col("x")), &s));
        assert!(may_match(&lit(20i64).lt_eq(col("x")), &s));
        assert!(!may_match(&col("x").gt(lit(20i64)), &s));
        assert!(may_match(&col("x").gt_eq(lit(20i64)), &s));
        let singleton = stats(DataType::Int64, "10", "10");
        assert!(!may_match(&col("x").neq(lit(10i64)), &singleton));
        assert!(may_match(&col("x").neq(lit(11i64)), &singleton));
    }

    #[test]
    fn bounds_are_conservative_and_numeric() {
        let s = stats(DataType::String, "abc", "abd");
        assert!(may_match(&col("x").eq(lit("abc")), &s));
        assert!(may_match(&col("x").neq(lit("abc")), &s));
        assert!(!may_match(&col("x").eq(lit("abe")), &s));
        let s = stats(DataType::Int64, "9007199254740993", "9007199254740993");
        assert!(!may_match(&col("x").eq(lit(9007199254740992i64)), &s));
        assert!(may_match(&col("x").eq(lit(9007199254740993i64)), &s));
        let s = stats(
            DataType::UInt64,
            "18446744073709551614",
            "18446744073709551615",
        );
        assert!(!may_match(&col("x").lt(lit(18446744073709551614u64)), &s));
    }

    #[test]
    fn absent_invalid_or_incompatible_statistics_are_unknown() {
        let expr = col("x").eq(lit(100i64));
        assert!(may_match(&expr, &HashMap::new()));
        for (min, max) in [("", "20"), ("20", "10"), ("999999999999999999999999", "20")] {
            assert!(may_match(&expr, &stats(DataType::Int64, min, max)));
        }
        let mut s = stats(DataType::Int64, "10", "20");
        s.get_mut("x").unwrap().min = None;
        assert!(may_match(&expr, &s));
        s.get_mut("x").unwrap().min = Some("10".into());
        s.get_mut("x").unwrap().null_count = Some(-1);
        assert!(may_match(&expr, &s));
        let s = stats(DataType::Int64, "10", "20");
        // Cross-type coercion may round large integers; do not guess its result.
        assert!(may_match(&col("x").eq(lit(100.0f64)), &s));
    }

    #[test]
    fn null_counts_are_separate_from_non_null_counts() {
        let mut s = stats(DataType::Int64, "10", "20");
        assert!(!may_match(&col("x").is_null(), &s));
        assert!(may_match(&col("x").is_not_null(), &s));
        s.get_mut("x").unwrap().null_count = Some(5);
        s.get_mut("x").unwrap().value_count = Some(0);
        assert!(may_match(&col("x").is_null(), &s));
        assert!(!may_match(&col("x").is_not_null(), &s));
        assert!(!may_match(&col("x").eq(lit(10i64)), &s));
        s.get_mut("x").unwrap().value_count = None;
        assert!(may_match(&col("x").is_not_null(), &s));
        assert!(!may_match(&col("x").eq(lit(NULL)), &s));
        s.get_mut("x").unwrap().dtype = DataType::List(Box::new(DataType::Int64));
        s.get_mut("x").unwrap().null_count = Some(0);
        assert!(may_match(&col("x").is_null(), &s));
    }

    #[test]
    fn nan_and_float_bounds() {
        let mut s = stats(DataType::Float64, "1.0", "2.0");
        assert!(!may_match(&col("x").gt(lit(2.0f64)), &s));
        s.get_mut("x").unwrap().contains_nan = Some(true);
        assert!(may_match(&col("x").gt(lit(2.0f64)), &s));
        assert!(may_match(&col("x").eq(lit(f64::NAN)), &s));
        s.get_mut("x").unwrap().contains_nan = None;
        assert!(may_match(&col("x").neq(lit(1.0f64)), &s));
        s.get_mut("x").unwrap().contains_nan = Some(false);
        s.get_mut("x").unwrap().min = Some("NaN".into());
        assert!(may_match(&col("x").gt(lit(2.0f64)), &s));
    }

    #[test]
    fn boolean_combinations_with_unknown_expressions() {
        let s = stats(DataType::Int64, "10", "20");
        let impossible = col("x").gt(lit(20i64));
        let unknown = col("y").is_null();
        assert!(!may_match(&(impossible.clone().and(unknown.clone())), &s));
        assert!(may_match(&(impossible.or(unknown)), &s));
        assert!(may_match(&col("x").gt(lit(20i64)).not(), &s));
    }

    #[test]
    fn range_membership_and_nan_excluding_bounds() {
        let s = stats(DataType::Int64, "1", "5");
        assert!(!may_match(
            &col("x").is_between(lit(10i64), lit(20i64), ClosedInterval::Both),
            &s
        ));
        assert!(!may_match(
            &col("x").is_in(lit(Series::new("".into(), [10i64, 20])), false),
            &s
        ));
        assert!(may_match(
            &col("x").is_in(lit(Series::new("".into(), [5i64, 20])), false),
            &s
        ));
        let mut s = stats(DataType::Float64, "10", "20");
        s.get_mut("x").unwrap().contains_nan = None;
        assert!(!may_match(&col("x").lt(lit(5.0f64)), &s));
        assert!(!may_match(&col("x").eq(lit(5.0f64)), &s));
        assert!(may_match(&col("x").gt(lit(25.0f64)), &s));
    }

    #[test]
    fn typed_temporal_and_decimal_bounds() {
        let s = stats(DataType::Date, "2024-01-01", "2024-01-31");
        let date = Expr::Literal(LiteralValue::Scalar(Scalar::new(
            DataType::Date,
            AnyValue::Date(0),
        )));
        assert!(!may_match(&col("x").eq(date), &s));
        let s = stats(DataType::Decimal(12, 2), "10.00", "20.00");
        let decimal = Expr::Literal(LiteralValue::Scalar(Scalar::new(
            DataType::Decimal(12, 2),
            AnyValue::Decimal(2500, 12, 2),
        )));
        assert!(!may_match(&col("x").eq(decimal), &s));
        assert!(!may_match(
            &col("x").eq(lit(true)),
            &stats(DataType::Boolean, "0", "0")
        ));
    }
}
