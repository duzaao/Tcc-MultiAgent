project = "agent-ai-os"
region = "us-central1"
zone = "us-central1-a"
instance_name = "agent-deploy-vm"
# CPU-only profile sized for deepseek-r1:8b
machine_type = "custom-8-32768" # 8 vCPUs, 32 GB RAM
boot_disk_size_gb = 200

# Disable GPU — CPU-only
enable_gpu = false
gpu_type = ""
gpu_count = 0

# Optional: repo to clone at startup (leave empty to scp the deploy folder manually)
repo_url = ""

# Wait seconds for manual upload if repo_url is empty
wait_for_upload_seconds = 600


# Replace with your admin IP or CIDR (recommended: your workstation IP/32)
admin_cidrs = ["YOUR_IP/32"]
