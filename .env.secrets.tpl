# ==============================================================================
# T1D Analytics Suite - Secrets & Private Credentials Template
# Copy this file to .env.secrets and fill in your values:
#   cp .env.secrets.tpl .env.secrets
# Note: .env.secrets is git-ignored and must NEVER be committed.
# ==============================================================================

# Backend API & Dataset Authentication
T1D_API_KEY=
API_KEY=
T1D_AUTH_TOKEN=

# LLM Provider API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
GOOGLE_API_KEY=
HF_TOKEN=

# Cloud & Remote Host Account Credentials
GCP_PROJECT_ID=
PRIMARY_CLOUD_STORAGE_ACCOUNT=your-storage-account
REMOTE_USER=azureuser
REMOTE_GROUP=azureuser
REMOTE_HOME=/home/azureuser
REMOTE_APP_DIR=/home/azureuser/t1d-analytics
