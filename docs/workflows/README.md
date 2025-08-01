# GitHub Actions Workflows Documentation

This directory contains documentation for GitHub Actions workflows that should be implemented for the retro-peft-adapters project. Since automated workflow creation is restricted, these serve as templates and specifications for manual implementation.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and security checks on every push and PR.

**Triggers**:
- Push to main branch
- Pull requests to main branch
- Scheduled daily runs for dependency security checks

**Jobs**:

#### Test Matrix
```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
    os: [ubuntu-latest, windows-latest, macos-latest]
```

#### Steps:
1. **Setup**: Checkout code, setup Python
2. **Dependencies**: Install package with test dependencies
3. **Linting**: Run black, isort, flake8, mypy
4. **Security**: Run bandit, safety checks
5. **Testing**: Run pytest with coverage
6. **Coverage**: Upload coverage to codecov

#### Performance Gates:
- Test coverage must be ≥80%
- All linting checks must pass
- Security scans must pass
- Performance regression tests

### 2. Security Scanning (`security.yml`)

**Purpose**: Comprehensive security analysis including dependency vulnerabilities, SAST, and supply chain security.

**Triggers**:
- Push to main branch
- Pull requests
- Scheduled weekly scans
- Manual dispatch

**Security Tools**:
1. **CodeQL** - Static application security testing
2. **Dependency Scanning** - Known vulnerability detection
3. **Secrets Scanning** - Prevent credential commits
4. **SBOM Generation** - Software Bill of Materials
5. **Container Scanning** - If Docker images are built

#### SLSA Compliance:
- Generate SLSA Level 2+ provenance
- Sign artifacts with sigstore
- Verify supply chain integrity

### 3. Release Automation (`release.yml`)

**Purpose**: Automated releases when version tags are pushed.

**Triggers**:
- Tag push matching `v*.*.*` pattern

**Release Process**:
1. **Validation**: Verify tag format and changelog
2. **Build**: Create distribution packages (wheel, sdist)
3. **Testing**: Full test suite on built packages
4. **Security**: Final security scans
5. **Publish**: Upload to PyPI (test first, then production)
6. **Documentation**: Update docs and generate release notes
7. **Notifications**: Slack/Discord/email notifications

#### Release Artifacts:
- Python wheel and source distribution
- SBOM (Software Bill of Materials)  
- Signed checksums
- Release notes
- Updated documentation

### 4. Documentation (`docs.yml`)

**Purpose**: Build and deploy documentation automatically.

**Triggers**:
- Push to main branch (docs changes)
- Pull requests affecting docs/
- Manual dispatch

**Documentation Stack**:
- **Sphinx** - API documentation generation
- **ReadTheDocs Theme** - Consistent styling
- **Auto-API** - Automatic API docs from docstrings
- **Jupyter Notebooks** - Example tutorials
- **Benchmarks** - Performance documentation

#### Deployment:
- GitHub Pages for development docs
- ReadTheDocs for stable releases
- Versioned documentation

### 5. Benchmarking (`benchmark.yml`)

**Purpose**: Performance regression testing and benchmarking.

**Triggers**:
- Push to main branch
- Pull requests with performance-critical changes
- Scheduled weekly runs
- Manual dispatch

**Benchmark Types**:
1. **Training Speed** - Adapter training performance
2. **Inference Latency** - Generation speed tests  
3. **Memory Usage** - Memory profiling
4. **Retrieval Performance** - Vector search speed
5. **Model Size** - Adapter parameter efficiency

#### Performance Tracking:
- Store benchmark results in database
- Generate performance trend reports
- Alert on significant regressions (>5%)
- Compare against baseline models

### 6. Dependency Updates (`deps.yml`)

**Purpose**: Automated dependency management and security updates.

**Triggers**:
- Scheduled weekly runs
- Manual dispatch
- Security advisory webhooks

**Update Strategy**:
1. **Security Updates** - Immediate automated PRs
2. **Minor Updates** - Weekly batch updates  
3. **Major Updates** - Manual review required
4. **Test Compatibility** - Full test suite on updates
5. **Rollback** - Automatic revert on test failures

#### Tools Integration:
- Dependabot for dependency scanning
- Safety for security advisories
- Pip-audit for vulnerability detection
- Custom scripts for ML-specific dependencies

## Workflow Configuration

### Environment Variables
```yaml
env:
  PYTHON_VERSION: '3.9'
  PYTORCH_VERSION: '2.0.0'
  CUDA_VERSION: '11.8'
  COVERAGE_THRESHOLD: '80'
```

### Secrets Required
- `PYPI_TOKEN` - PyPI publishing
- `CODECOV_TOKEN` - Coverage reporting  
- `SLACK_WEBHOOK` - Notifications
- `GPG_PRIVATE_KEY` - Artifact signing

### Caching Strategy
- **pip cache** - Python dependencies
- **model cache** - Pre-trained models for testing
- **build cache** - Compiled extensions
- **test cache** - Test data and fixtures

## Implementation Checklist

- [ ] Create `.github/workflows/` directory
- [ ] Implement CI workflow with test matrix
- [ ] Set up security scanning with CodeQL
- [ ] Configure release automation
- [ ] Set up documentation building
- [ ] Implement benchmarking pipeline
- [ ] Configure dependency update automation
- [ ] Set up required secrets and environment variables
- [ ] Test all workflows in feature branch
- [ ] Enable branch protection rules
- [ ] Configure notification channels

## Branch Protection Rules

Configure the following branch protection rules for `main`:

1. **Require status checks**: All CI jobs must pass
2. **Require up-to-date branches**: Force rebase/merge
3. **Require signed commits**: Enhanced security
4. **Dismiss stale reviews**: On new commits
5. **Require admin review**: For workflow changes
6. **Restrict pushes**: Only through PRs

## Performance Targets

### CI Pipeline Performance:
- **Total CI time**: <15 minutes
- **Test execution**: <10 minutes
- **Linting**: <2 minutes
- **Security scans**: <5 minutes

### Resource Optimization:
- Parallel job execution
- Selective test running based on changes
- Artifact caching and reuse
- Matrix job failure fast-fail

## Monitoring and Alerting

### Metrics to Track:
- CI success rate (target: >95%)
- Average pipeline duration
- Test flakiness rate
- Security scan findings
- Dependency update frequency

### Alert Conditions:
- CI failure rate >5%
- Security vulnerabilities detected
- Performance regression >10%
- Test coverage drop below threshold
- Dependency updates blocked >1 week

This documentation provides comprehensive guidelines for implementing robust CI/CD workflows while maintaining security and performance standards.