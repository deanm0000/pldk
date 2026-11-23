use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

use object_store::azure::MicrosoftAzureBuilder;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use polars::prelude::*;
use url::Url;

use crate::Result;

pub(crate) struct Storage {
    key: String,
    stores: Mutex<HashMap<(String, String), Arc<dyn ObjectStore>>>,
}

impl Storage {
    pub fn new(key: String) -> Self {
        Self {
            key,
            stores: Mutex::new(HashMap::new()),
        }
    }

    fn location(&self, path: &str) -> Result<(Arc<dyn ObjectStore>, Path)> {
        let url = Url::parse(path).map_err(|_| "Expected an absolute Azure blob URL")?;
        if url.query().is_some() || url.fragment().is_some() {
            return Err("Catalog paths must not embed credentials or fragments".into());
        }
        let (account, container, blob) = match url.scheme() {
            "https" => {
                let host = url.host_str().ok_or("Missing Azure storage host")?;
                let account = host
                    .strip_suffix(".blob.core.windows.net")
                    .or_else(|| host.strip_suffix(".dfs.core.windows.net"))
                    .ok_or("Only Azure public-cloud blob endpoints are supported")?;
                let (container, blob) = url
                    .path()
                    .trim_start_matches('/')
                    .split_once('/')
                    .ok_or("Missing blob path")?;
                (account.to_owned(), container.to_owned(), blob.to_owned())
            }
            "az" | "azure" => {
                let container = url.host_str().ok_or("Missing Azure container")?;
                let account = if let Ok(credential) = Url::parse(&self.key) {
                    credential
                        .host_str()
                        .and_then(|h| h.strip_suffix(".blob.core.windows.net"))
                        .map(str::to_owned)
                } else {
                    self.key
                        .split(';')
                        .filter_map(|s| s.split_once('='))
                        .find(|(k, _)| *k == "AccountName")
                        .map(|(_, v)| v.to_owned())
                }
                .ok_or(
                    "az:// paths require a SAS URL or connection string containing the account",
                )?;
                (
                    account,
                    container.to_owned(),
                    url.path().trim_start_matches('/').to_owned(),
                )
            }
            "abfs" | "abfss" => {
                let account = url
                    .host_str()
                    .and_then(|h| h.strip_suffix(".dfs.core.windows.net"))
                    .ok_or("Expected an Azure Data Lake Storage endpoint")?;
                if url.username().is_empty() {
                    return Err("Missing Azure container in abfs:// URL".into());
                }
                (
                    account.to_owned(),
                    url.username().to_owned(),
                    url.path().trim_start_matches('/').to_owned(),
                )
            }
            // TODO(storage): add other object_store providers when supported.
            _ => return Err("Only Azure object storage is supported".into()),
        };
        let blob = Path::from_url_path(&blob).map_err(|_| "Invalid Azure blob path")?;
        let cache_key = (account.clone(), container.clone());
        if let Some(store) = self
            .stores
            .lock()
            .map_err(|_| "Azure client cache lock poisoned")?
            .get(&cache_key)
        {
            return Ok((store.clone(), blob));
        }
        let mut builder = MicrosoftAzureBuilder::new()
            .with_account(&account)
            .with_container_name(&container);
        if self.key.contains("AccountName=") {
            let parts: std::collections::HashMap<_, _> = self
                .key
                .split(';')
                .filter_map(|s| s.split_once('='))
                .collect();
            if parts.get("AccountName").copied() != Some(account.as_str()) {
                return Err("Azure connection string account does not match catalog path".into());
            }
            if let Some(key) = parts.get("AccountKey") {
                builder = builder.with_access_key(*key);
            } else if let Some(sas) = parts.get("SharedAccessSignature") {
                builder = builder.with_sas_authorization(
                    url::form_urlencoded::parse(sas.as_bytes())
                        .into_owned()
                        .collect::<Vec<_>>(),
                );
            } else {
                return Err(
                    "Azure connection string has no AccountKey or SharedAccessSignature".into(),
                );
            }
        } else if let Ok(credential) = Url::parse(&self.key) {
            if credential.scheme() != "https"
                || credential.host_str()
                    != Some(format!("{account}.blob.core.windows.net").as_str())
                || credential.path().trim_matches('/') != container
            {
                return Err("SAS URL does not match the catalog storage account/container".into());
            }
            let query = credential.query().ok_or("SAS URL has no token")?;
            builder = builder.with_sas_authorization(
                url::form_urlencoded::parse(query.as_bytes())
                    .into_owned()
                    .collect::<Vec<_>>(),
            );
        } else if self.key.trim_start_matches('?').contains("sig=") {
            builder = builder.with_sas_authorization(
                url::form_urlencoded::parse(self.key.trim_start_matches('?').as_bytes())
                    .into_owned()
                    .collect::<Vec<_>>(),
            );
        } else {
            builder = builder.with_access_key(&self.key);
        }
        let store = builder
            .build()
            .map_err(|_| "Cannot configure Azure object store")?;
        let store: Arc<dyn ObjectStore> = Arc::new(store);
        self.stores
            .lock()
            .map_err(|_| "Azure client cache lock poisoned")?
            .insert(cache_key, store.clone());
        Ok((store, blob))
    }

    pub async fn parquet(&self, path: &str) -> Result<ParquetReader<Cursor<Vec<u8>>>> {
        let (store, path) = self.location(path)?;
        let bytes = store
            .get(&path)
            .await
            .map_err(|_| "Azure blob download failed (check credentials and blob existence)")?
            .bytes()
            .await
            .map_err(|_| "Azure blob download was interrupted")?;
        Ok(ParquetReader::new(Cursor::new(bytes.to_vec())))
    }
}

pub(crate) fn resolve(base: &str, path: &str, relative: bool) -> String {
    if relative {
        format!(
            "{}/{}",
            base.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    } else {
        path.to_owned()
    }
}
