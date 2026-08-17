# Models

# Model files

## SAM3

Text-prompt mode requires `models/sam3.pt`. It is intentionally ignored by Git because
the file is about 3.45 GB, far above GitHub's 100 MB per-file limit. Copy it locally:

```bash
cp /path/to/sam3.pt models/sam3.pt
```

The locally verified checkpoint has SHA-256:

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

You may store it elsewhere and set `SAM3_MODEL_PATH=/absolute/path/to/sam3.pt`.

## MobileSAM

When the text prompt is empty, the legacy template/background workflow uses the
bundled `mobile_sam.pt`. Its SHA-256 checksum is:

```text
6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
```

