use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use stdlib::Target;

use crate::project_manifest::{
    GoDependency, GoReplacement, Manifest, check_go_replacements, check_no_subpackage_deps,
    check_toolchain_version, find_module_for_pkg, go_cache_version, go_replacement_cache_key,
    parse_manifest,
};
use crate::{GoModule, GoPackage, typedef_cache_dir};

#[derive(Debug)]
pub enum TypedefLocatorResult {
    Found {
        content: Cow<'static, str>,
        origin: TypedefOrigin,
    },
    /// Looks like a stdlib package but no stdlib typedef exists.
    UnknownStdlib,
    /// Has a domain-style path but is not declared in the manifest.
    UndeclaredImport,
    /// Declared in the manifest but no `.d.lis` on disk and no bindgen runner.
    MissingTypedef { module: String, version: String },
    /// Typedef file exists but could not be read.
    UnreadableTypedef { path: PathBuf, error: String },
    /// The bindgen runner ran on cache miss but failed.
    BindgenFailed {
        module: String,
        version: String,
        package: String,
        kind: BindgenFailure,
    },
}

/// Why a `Bindgen::run` invocation failed.
#[derive(Debug)]
pub enum BindgenFailure {
    /// `go` is not installed or not on PATH.
    GoToolchainMissing,
    /// The bindgen subprocess failed; `stderr` is the trimmed message.
    InvocationFailed { stderr: String },
}

/// Classification of a `go:` import path without touching the cache.
#[derive(Debug)]
pub enum DeclarationStatus {
    Stdlib,
    DeclaredThirdParty {
        module: String,
        version: String,
        cache_version: String,
    },
    UnknownStdlib,
    UndeclaredImport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedefOrigin {
    Stdlib,
    Cache(PathBuf),
}

impl TypedefOrigin {
    /// Consume the origin, yielding the on-disk path for cache origins and
    /// `None` for embedded stdlib typedefs.
    pub fn into_cache_path(self) -> Option<PathBuf> {
        match self {
            TypedefOrigin::Cache(path) => Some(path),
            TypedefOrigin::Stdlib => None,
        }
    }
}

/// Cache-miss hook the locator invokes for declared third-party packages.
pub trait Bindgen: Send + Sync + std::fmt::Debug {
    /// Generate `pkg`'s typedef and write it to the cache (no transitives).
    fn run(&self, pkg: &GoPackage) -> Result<(), BindgenFailure>;
}

/// Per-analysis bindgen runner; dropping the session releases the lock.
pub struct BindgenSession {
    pub bindgen: std::sync::Arc<dyn Bindgen>,
    _guard: Box<dyn BindgenGuard>,
}

impl BindgenSession {
    pub fn new(bindgen: std::sync::Arc<dyn Bindgen>, guard: Box<dyn BindgenGuard>) -> Self {
        Self {
            bindgen,
            _guard: guard,
        }
    }
}

pub trait BindgenGuard: sealed::Sealed + Send {}

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for std::fs::File {}
impl BindgenGuard for std::fs::File {}

pub trait BindgenSetup: Send + Sync {
    fn for_project(&self, project_root: &Path, target: Target) -> Result<BindgenSession, String>;
}

#[derive(Debug, Clone, Default)]
pub struct TypedefLocator {
    deps: BTreeMap<String, GoDependency>,
    replacements: BTreeMap<String, GoReplacement>,
    replacement_cache_keys: BTreeMap<String, String>,
    project_root: Option<PathBuf>,
    target: Target,
    bindgen: Option<Arc<dyn Bindgen>>,
}

impl TypedefLocator {
    pub fn new(
        deps: BTreeMap<String, GoDependency>,
        project_root: Option<PathBuf>,
        target: Target,
    ) -> Self {
        Self::new_with_replacements(deps, BTreeMap::new(), project_root, target)
    }

    pub fn new_with_replacements(
        deps: BTreeMap<String, GoDependency>,
        replacements: BTreeMap<String, GoReplacement>,
        project_root: Option<PathBuf>,
        target: Target,
    ) -> Self {
        let replacement_cache_keys = replacements
            .iter()
            .map(|(module, replacement)| {
                (
                    module.clone(),
                    go_replacement_cache_key(replacement, project_root.as_deref()),
                )
            })
            .collect();

        Self {
            deps,
            replacements,
            replacement_cache_keys,
            project_root,
            target,
            bindgen: None,
        }
    }

    pub fn from_project(project_root: &Path) -> Result<Self, String> {
        let (_, locator) = Self::from_project_with_manifest(project_root)?;
        Ok(locator)
    }

    pub fn from_project_with_manifest(project_root: &Path) -> Result<(Manifest, Self), String> {
        let manifest = parse_manifest(project_root)?;

        check_toolchain_version(&manifest)?;
        check_no_subpackage_deps(&manifest)?;
        check_go_replacements(&manifest)?;

        let project_root = project_root.canonicalize().map_err(|e| {
            format!(
                "Failed to resolve project path `{}`: {}",
                project_root.display(),
                e
            )
        })?;

        let locator = Self::new_with_replacements(
            manifest.go_deps(),
            manifest.go_replacements(),
            Some(project_root),
            Target::host(),
        );

        Ok((manifest, locator))
    }

    pub fn with_bindgen(mut self, bindgen: Arc<dyn Bindgen>) -> Self {
        self.bindgen = Some(bindgen);
        self
    }

    pub fn project_root(&self) -> Option<&Path> {
        self.project_root.as_deref()
    }

    pub fn deps(&self) -> &BTreeMap<String, GoDependency> {
        &self.deps
    }

    pub fn replacements(&self) -> &BTreeMap<String, GoReplacement> {
        &self.replacements
    }

    pub fn target(&self) -> Target {
        self.target
    }

    pub fn is_declared_go_dep(&self, package_path: &str) -> bool {
        find_module_for_pkg(&self.deps, package_path).is_some()
    }

    /// Resolve a `go:` package path to its declared module path and version by
    /// longest declared prefix, or `None` if no declared module contains it.
    pub fn module_for_package(&self, package_path: &str) -> Option<(String, String, String)> {
        find_module_for_pkg(&self.deps, package_path).map(|(module, dep)| {
            (
                module.to_string(),
                dep.version.clone(),
                self.cache_version(module, dep),
            )
        })
    }

    /// Classify a `go:` import path without touching the cache or bindgen.
    pub fn validate_declaration(&self, package_path: &str) -> DeclarationStatus {
        self.classify(package_path)
    }

    fn classify(&self, package_path: &str) -> DeclarationStatus {
        if crate::is_stdlib(package_path) {
            return match stdlib::get_go_stdlib_typedef(package_path, self.target) {
                Some(_) => DeclarationStatus::Stdlib,
                None => DeclarationStatus::UnknownStdlib,
            };
        }

        match find_module_for_pkg(&self.deps, package_path) {
            Some((module_path, dep)) => DeclarationStatus::DeclaredThirdParty {
                module: module_path.to_string(),
                version: dep.version.clone(),
                cache_version: self.cache_version(module_path, dep),
            },
            None => DeclarationStatus::UndeclaredImport,
        }
    }

    fn cache_version(&self, module_path: &str, dep: &GoDependency) -> String {
        go_cache_version(
            &dep.version,
            self.replacement_cache_keys
                .get(module_path)
                .map(String::as_str),
        )
    }

    /// Resolve a `go:` package: stdlib -> on-disk cache -> bindgen runner if set.
    pub fn find_typedef_content(&self, package_path: &str) -> TypedefLocatorResult {
        let (module_path, version, cache_version) = match self.classify(package_path) {
            DeclarationStatus::Stdlib => {
                let source = stdlib::get_go_stdlib_typedef(package_path, self.target)
                    .expect("Stdlib classification implies an embedded typedef");
                // Record the path the editor will navigate to; the files are
                // written by the LSP at startup. No HOME -> non-navigable.
                let origin = match crate::stdlib_typedef_path(self.target, package_path) {
                    Some(path) => TypedefOrigin::Cache(path),
                    None => TypedefOrigin::Stdlib,
                };
                return TypedefLocatorResult::Found {
                    content: Cow::Borrowed(source),
                    origin,
                };
            }
            DeclarationStatus::UnknownStdlib => return TypedefLocatorResult::UnknownStdlib,
            DeclarationStatus::UndeclaredImport => return TypedefLocatorResult::UndeclaredImport,
            DeclarationStatus::DeclaredThirdParty {
                module,
                version,
                cache_version,
            } => (module, version, cache_version),
        };

        let Some(project_root) = &self.project_root else {
            return TypedefLocatorResult::MissingTypedef {
                module: module_path,
                version,
            };
        };

        let pkg = GoPackage {
            module: GoModule {
                path: &module_path,
                version: &version,
                cache_version: Some(&cache_version),
            },
            package: package_path,
        };
        let cache_dir = typedef_cache_dir(project_root);
        let typedef_path = pkg.typedef_path(&cache_dir, self.target);

        let read_outcome =
            if self.local_replacement_typedef_stale(&module_path, package_path, &typedef_path) {
                ReadOutcome::Missing
            } else {
                read_typedef(&typedef_path)
            };

        match read_outcome {
            ReadOutcome::Found(content) => TypedefLocatorResult::Found {
                content: Cow::Owned(content),
                origin: TypedefOrigin::Cache(typedef_path),
            },
            ReadOutcome::Unreadable(error) => TypedefLocatorResult::UnreadableTypedef {
                path: typedef_path,
                error,
            },
            ReadOutcome::Missing => match &self.bindgen {
                None => TypedefLocatorResult::MissingTypedef {
                    module: module_path,
                    version,
                },
                Some(runner) => match runner.run(&pkg) {
                    Ok(()) => match read_typedef(&typedef_path) {
                        ReadOutcome::Found(content) => TypedefLocatorResult::Found {
                            content: Cow::Owned(content),
                            origin: TypedefOrigin::Cache(typedef_path),
                        },
                        ReadOutcome::Unreadable(error) => TypedefLocatorResult::UnreadableTypedef {
                            path: typedef_path,
                            error,
                        },
                        ReadOutcome::Missing => TypedefLocatorResult::BindgenFailed {
                            module: module_path,
                            version,
                            package: package_path.to_string(),
                            kind: BindgenFailure::InvocationFailed {
                                stderr: format!(
                                    "bindgen reported success but `{}` was not written",
                                    typedef_path.display()
                                ),
                            },
                        },
                    },
                    Err(kind) => TypedefLocatorResult::BindgenFailed {
                        module: module_path,
                        version,
                        package: package_path.to_string(),
                        kind,
                    },
                },
            },
        }
    }

    fn local_replacement_typedef_stale(
        &self,
        module_path: &str,
        package_path: &str,
        typedef_path: &Path,
    ) -> bool {
        let Some(replacement) = self.replacements.get(module_path) else {
            return false;
        };
        let Some(path) = &replacement.path else {
            return false;
        };
        let Some(package_rel) = package_path
            .strip_prefix(module_path)
            .filter(|rest| rest.is_empty() || rest.starts_with('/'))
        else {
            return false;
        };

        let root = crate::resolve_go_replacement_path(path, self.project_root.as_deref());
        let package_dir = root.join(package_rel.trim_start_matches('/'));
        let Some(package_mtime) = go_package_mtime_nanos(&package_dir) else {
            return false;
        };
        let Some(typedef_mtime) = file_mtime_nanos(typedef_path) else {
            return true;
        };

        package_mtime > typedef_mtime
    }
}

fn go_package_mtime_nanos(package_dir: &Path) -> Option<u128> {
    let entries = std::fs::read_dir(package_dir).ok()?;
    let mut max = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("go") {
            continue;
        }
        max = max.max(file_mtime_nanos(&path));
    }
    max
}

fn file_mtime_nanos(path: &Path) -> Option<u128> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    system_time_nanos(modified)
}

fn system_time_nanos(time: SystemTime) -> Option<u128> {
    time.duration_since(UNIX_EPOCH).ok().map(|d| d.as_nanos())
}

enum ReadOutcome {
    Found(String),
    Missing,
    Unreadable(String),
}

fn read_typedef(path: &Path) -> ReadOutcome {
    match std::fs::read_to_string(path) {
        Ok(s) => ReadOutcome::Found(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ReadOutcome::Missing,
        Err(e) => ReadOutcome::Unreadable(e.to_string()),
    }
}
