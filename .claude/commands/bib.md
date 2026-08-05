---
description: Fetch a canonical BibTeX entry from a loose citation.
---

Use the `reference-manager` agent to fetch a canonical BibTeX entry for:

$ARGUMENTS

Use zbMATH or the publisher's DOI page as the primary source. Avoid Google
Scholar BibTeX export (frequently wrong). Normalize the cite key following the
project convention (`LastnameYYYYKeyword`). Check existing `.bib` files in
`references/` and the active manuscript for duplicates before adding. Return
the cite key to use plus the full entry added, and report which `.bib` file
received it.
