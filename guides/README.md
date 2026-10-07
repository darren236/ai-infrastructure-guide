# Runnable guide assets

The [written guides](../docs/guides/README.md) contain instructions and recorded observations. This directory is reserved for reusable code and configuration created as those guides progress.

When a guide needs files, use a directory matching its document name:

```text
guides/
├── 01-brev-gpu-node-validation/              # Future code/configuration
└── 02-nemotron-streaming-singapore-english/  # Future code/configuration
```

These subdirectories have not been created yet; commands currently live in the written guides. Add reusable files when an actual guide step needs them, and link to them from the written guide. Container definitions, scheduler scripts, and manifests belong with the workload they support. Add shared tools elsewhere only when multiple guides use them.

Keep downloaded audio, datasets, model caches, credentials, raw recordings, and raw workload outputs outside the repository. [Return to the repository overview](../README.md).
