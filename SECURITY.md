# Security Policy

## Supported versions

vanguarstew is pre-1.0 and under active development. Security fixes are applied to the
latest `main` only.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's private vulnerability reporting: open the repository's **Security**
tab and choose **Report a vulnerability**. If that is unavailable, email
`iot2edge7@outlook.com` with the details.

Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal repro or command line is ideal),
- affected files or components, and
- any suggested remediation.

We aim to acknowledge reports within a few days and to keep you updated as we work on a
fix. Please give us a reasonable window to remediate before any public disclosure.

## Scope notes

vanguarstew executes agent code and reads untrusted repositories during evaluation.
Reports involving sandbox escape, inference-proxy abuse, task/answer leakage, or
judge prompt-injection are especially valuable.
