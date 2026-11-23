use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use polars::prelude::*;

use crate::Result;

#[derive(Clone)]
pub(crate) struct Column {
    pub id: i64,
    pub name: String,
    pub dtype: DataType,
    pub initial_default: Option<String>,
    pub bytea_text: bool,
}

pub(crate) fn decode_hex(value: &str) -> Result<Vec<u8>> {
    if !value.is_ascii() || value.len() % 2 != 0 {
        return Err("Invalid hex-encoded inline value".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&value[i..i + 2], 16)
                .map_err(|_| "Invalid hex-encoded inline value".into())
        })
        .collect()
}

pub(crate) fn dtype(name: &str) -> Result<DataType> {
    Ok(match name {
        "boolean" => DataType::Boolean,
        "null" => DataType::Null,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "int128" => DataType::Int128,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "varchar" | "json" | "uuid" => DataType::String,
        "blob" => DataType::Binary,
        "date" => DataType::Date,
        "time" | "time_ns" => DataType::Time,
        "timestamp" => DataType::Datetime(TimeUnit::Microseconds, None),
        "timestamp_ms" => DataType::Datetime(TimeUnit::Milliseconds, None),
        "timestamp_ns" => DataType::Datetime(TimeUnit::Nanoseconds, None),
        "timestamptz" => DataType::Datetime(TimeUnit::Microseconds, Some(TimeZone::UTC)),
        "timestamptz_ns" => DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC)),
        "timestamp_s" => DataType::Datetime(TimeUnit::Milliseconds, None),
        s if s.starts_with("decimal(") => {
            let parts: Vec<_> = s
                .trim_start_matches("decimal(")
                .trim_end_matches(')')
                .split(',')
                .collect();
            if parts.len() != 2 {
                return Err(format!("Invalid DuckLake decimal type: {s}"));
            }
            let precision = parts[0]
                .trim()
                .parse::<usize>()
                .map_err(|e| e.to_string())?;
            let scale = parts[1]
                .trim()
                .parse::<usize>()
                .map_err(|e| e.to_string())?;
            if !(1..=38).contains(&precision) || scale > precision {
                return Err(format!("Unsupported DuckLake decimal precision/scale: {s}"));
            }
            DataType::Decimal(precision, scale)
        }
        other => return Err(format!("Unsupported DuckLake type: {other}")),
    })
}

pub(crate) fn text_series(column: &Column, values: &[Option<String>]) -> Result<Series> {
    let name = column.name.clone().into();
    let invalid = |value: &str| format!("Invalid value for column {}: {value}", column.name);
    let series = match &column.dtype {
        DataType::Date => {
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let values = values
                .iter()
                .map(|s| {
                    s.as_deref()
                        .map(|s| {
                            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                                .map(|d| (d - epoch).num_days() as i32)
                                .map_err(|_| invalid("date"))
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Series::new(name, values)
        }
        DataType::Datetime(unit, zone) => {
            let values = values
                .iter()
                .map(|s| {
                    s.as_deref()
                        .map(|s| {
                            let utc = if zone.is_some() {
                                DateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f%#z")
                                    .map(|d| d.naive_utc())
                                    .map_err(|_| invalid("timestamp with time zone"))?
                            } else {
                                NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
                                    .map_err(|_| invalid("timestamp"))?
                            }
                            .and_utc();
                            match unit {
                                TimeUnit::Milliseconds => Ok(utc.timestamp_millis()),
                                TimeUnit::Microseconds => Ok(utc.timestamp_micros()),
                                TimeUnit::Nanoseconds => utc
                                    .timestamp_nanos_opt()
                                    .ok_or_else(|| invalid("timestamp out of nanosecond range")),
                            }
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Series::new(name, values)
        }
        DataType::Time => {
            let values = values
                .iter()
                .map(|s| {
                    s.as_deref()
                        .map(|s| {
                            NaiveTime::parse_from_str(s, "%H:%M:%S%.f")
                                .map(|t| {
                                    i64::from(t.num_seconds_from_midnight()) * 1_000_000_000
                                        + i64::from(t.nanosecond())
                                })
                                .map_err(|_| invalid("time"))
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Series::new(name, values)
        }
        DataType::Boolean => {
            let values = values
                .iter()
                .map(|s| {
                    s.as_deref()
                        .map(|s| match s {
                            "true" | "t" | "1" => Ok(true),
                            "false" | "f" | "0" => Ok(false),
                            _ => Err(invalid("boolean")),
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            Series::new(name, values)
        }
        DataType::Binary => {
            let values = values
                .iter()
                .map(|s| {
                    s.as_deref()
                        .map(|s| decode_hex(s).map_err(|_| invalid("hex-encoded blob")))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<_> = values.iter().map(|v| v.as_deref()).collect();
            Series::new(name, refs)
        }
        _ => Series::new(name, values),
    };
    series.strict_cast(&column.dtype).map_err(|e| e.to_string())
}

pub(crate) fn empty_frame(columns: &[Column]) -> Result<DataFrame> {
    DataFrame::new_infer_height(
        columns
            .iter()
            .map(|c| Series::new_empty(c.name.clone().into(), &c.dtype).into_column())
            .collect(),
    )
    .map_err(|e| e.to_string())
}
