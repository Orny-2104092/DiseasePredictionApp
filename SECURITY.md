# Security Documentation

This document outlines security measures, best practices, and considerations for the Disease Prediction App.

## Overview

The Disease Prediction App is built with security in mind. This document describes implemented security features and recommendations for maintaining a secure application.

## Implemented Security Measures

### 1. Authentication & Authorization

- **JWT Token Authentication**: All protected endpoints require valid JWT tokens
- **Password Hashing**: Uses bcrypt_sha256 with automatic pre-hashing for long passwords
- **Token Expiration**: Tokens expire after 1 hour (configurable)
- **Secure Token Storage**: Tokens stored in localStorage with same-site restrictions

### 2. Input Validation

- **Pydantic Models**: All request bodies are validated using Pydantic models
- **Input Sanitization**: User inputs are sanitized to prevent injection attacks
- **Length Restrictions**: All inputs have maximum length limits
- **Dangerous Pattern Detection**: SQL injection and XSS patterns are blocked

### 3. API Security

- **CORS Configuration**: Strict origin whitelist, no wildcards
- **Rate Limiting**: Configurable rate limits per endpoint
- **Request Size Limits**: Maximum body sizes enforced
- **Secure Headers**: Security headers recommended for production

### 4. Data Protection

- **Path Validation**: Model and data files validated to prevent path traversal
- **Environment Variables**: Sensitive config stored in environment, not code
- **Database Security**: Prepared statements prevent SQL injection

### 5. Frontend Security

- **Content Security Policy**: Recommended for production deployment
- **XSS Prevention**: Input validation and output encoding
- **Secure API Calls**: Authorization headers with every request

## Security Checklist

### Development
- [ ] Use `.env` files for all sensitive configuration
- [ ] Never commit `.env` files or secrets to version control
- [ ] Run security scans before merging code
- [ ] Review security scan reports

### Production Deployment
- [ ] Generate strong JWT_SECRET (minimum 32 characters)
- [ ] Configure proper CORS origins (no `*` wildcards)
- [ ] Enable HTTPS/TLS
- [ ] Set up proper rate limiting
- [ ] Configure secure cookie settings
- [ ] Enable security headers (HSTS, X-Frame-Options, etc.)
- [ ] Set up monitoring and alerting for suspicious activity
- [ ] Regular security audits and penetration testing

## Security Scanning

The project includes automated security scanning via GitHub Actions:

- **Bandit**: Python security linter
- **Safety**: Python dependency vulnerability scanner
- **Semgrep**: Static analysis for security patterns
- **ESLint**: JavaScript security scanning
- **NPM Audit**: Node.js dependency vulnerabilities
- **Pip Audit**: Python dependency vulnerabilities

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `NEON_DATABASE_URL` | Yes | PostgreSQL connection string | `postgresql://user:pass@host/db` |
| `JWT_SECRET` | Yes | Secret for JWT signing | `your-32-char-secret` |
| `JWT_ALGORITHM` | No | JWT algorithm (default: HS256) | `HS256` |
| `JWT_EXPIRES_SECONDS` | No | Token lifetime (default: 3600) | `3600` |
| `MODEL_PATH` | Yes | Path to model file | `Project/model/disease_model.pkl` |
| `SYMPTOM_LIST_PATH` | Yes | Path to symptom list | `Project/model/symptom_list.pkl` |
| `DATA_DIR` | Yes | Data directory path | `Project/data` |
| `ALLOWED_ORIGINS` | Yes | CORS origins (comma-separated) | `http://localhost:3000` |

## Common Security Issues & Mitigations

### 1. SQL Injection
**Mitigation**: Using SQLAlchemy ORM with parameterized queries. Never concatenate user input into SQL strings.

### 2. XSS (Cross-Site Scripting)
**Mitigation**: Input validation, output encoding, and Content Security Policy headers.

### 3. CSRF (Cross-Site Request Forgery)
**Mitigation**: Same-site cookie attributes and CORS configuration.

### 4. Authentication Bypass
**Mitigation**: JWT token verification on every protected endpoint, proper error handling.

### 5. Path Traversal
**Mitigation**: Path validation before file access, restrict file operations to allowed directories.

### 6. Information Disclosure
**Mitigation**: Generic error messages, proper HTTP status codes, no stack traces in responses.

## Reporting Security Vulnerabilities

If you discover a security vulnerability in this project, please:

1. Do NOT disclose it publicly
2. Contact the maintainers privately
3. Provide detailed information about the vulnerability
4. Allow time for a fix before any public disclosure

## Security Best Practices for Users

1. **Strong Passwords**: Use long, complex passwords
2. **Token Security**: Don't share or expose your auth tokens
3. **HTTPS**: Always use HTTPS in production
4. **Regular Updates**: Keep the application updated
5. **Monitor Activity**: Review logs for suspicious activity

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [SQLAlchemy Security](https://docs.sqlalchemy.org/en/14/core/security.html)
- [JWT Best Practices](https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/)
