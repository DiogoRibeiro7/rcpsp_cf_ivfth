# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| main    | ✅                 |
| develop | ✅ (active branch) |
| others  | ❌                 |

Security fixes are released as part of the regular development cycle. Legacy tags are not patched.

## Reporting a Vulnerability

Please report security vulnerabilities through GitHub Security Advisories:

1. Navigate to <https://github.com/DiogoRibeiro7/rcpsp_cf_ivfth/security/advisories/new>.
2. Provide a clear description of the issue, including steps to reproduce, impact, and any suggested fixes.
3. Expect an initial response within **5 business days**. We may request additional information or a proof of concept.

Alternatively, you can email `security@diogoribeiro7.dev` if the advisory form is unavailable.

Please **do not** disclose vulnerabilities publicly until we confirm a fix has shipped and the coordinated disclosure date has passed.

## Preferred Languages

We prefer vulnerability reports in **English** or **Portuguese**.

## Security Testing Guidelines

- Do not run automated scanners against production deployments owned by others.
- Avoid tests that could disrupt availability (e.g., large-scale fuzzing, denial-of-service).
- Never access or exfiltrate data that you do not own.

Thank you for helping keep RCPSP-CF-IVFTH secure!
