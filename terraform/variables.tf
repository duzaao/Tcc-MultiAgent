variable "project" {
  description = "GCP project id"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "europe-west1-b"
}

variable "instance_name" {
  description = "Name of the compute instance"
  type        = string
  default     = "agent-deploy-vm"
}

variable "machine_type" {
  description = "Machine type for the instance"
  type        = string
  # Default to a larger CPU-only custom machine suitable for ~8B models (8 vCPUs, 32GB RAM)
  default     = "custom-8-32768"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "repo_url" {
  description = "Git repo URL to clone on startup (optional). If empty, upload the deploy folder manually."
  type        = string
  default     = ""
}

variable "enable_gpu" {
  description = "Whether to attach GPU to the instance"
  type        = bool
  default     = false
}

variable "gpu_type" {
  description = "GPU type to request (e.g. nvidia-tesla-t4, nvidia-tesla-a100)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 0
}

variable "admin_cidrs" {
  description = "CIDR ranges allowed to access agent/API ports and SSH. Use your admin IP(s)."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "wait_for_upload_seconds" {
  description = "When repo_url is empty, how long (seconds) the startup script will wait for a manual upload to /home/<user>/deploy"
  type        = number
  default     = 600
}
