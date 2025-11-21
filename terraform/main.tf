resource "google_compute_instance" "deploy_vm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
      size  = var.boot_disk_size_gb
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  scheduling {
  on_host_maintenance = "TERMINATE"
  automatic_restart    = false
  }


  metadata = {
    startup-script = <<-EOT
      #!/bin/bash
      set -e
      apt-get update
      apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release git
      # Install Docker
      curl -fsSL https://get.docker.com | sh
      # Install docker compose plugin (Debian package may vary)
      apt-get install -y docker-compose-plugin || true
      systemctl enable --now docker || true

      # If repo_url provided, clone and run compose. Otherwise wait for manual upload to ~/deploy
      REPO_URL="${var.repo_url}"
      TARGET_DIR="/opt/agent-deploy"
      USER_DEPLOY_DIR="/home/$(whoami)/deploy"
      WAIT_SECONDS=${var.wait_for_upload_seconds}

      if [ -n "$REPO_URL" ]; then
        rm -rf "$TARGET_DIR"
        git clone "$REPO_URL" "$TARGET_DIR" || (cd "$TARGET_DIR" && git pull)
        if [ -d "$TARGET_DIR/deploy/infra" ]; then
          cd "$TARGET_DIR/deploy/infra"
          docker compose up -d --build
        else
          echo "Cloned repo but did not find deploy/infra in $TARGET_DIR"
        fi
      else
        echo "No repo_url provided. Waiting up to $WAIT_SECONDS seconds for a manual upload to $USER_DEPLOY_DIR"
        elapsed=0
        interval=5
        while [ $elapsed -lt $WAIT_SECONDS ]; do
          if [ -d "$USER_DEPLOY_DIR" ]; then
            echo "Found $USER_DEPLOY_DIR — installing"
            rm -rf "$TARGET_DIR"
            mkdir -p "$TARGET_DIR"
            cp -a "$USER_DEPLOY_DIR/." "$TARGET_DIR/"
            if [ -d "$TARGET_DIR/deploy/infra" ]; then
              cd "$TARGET_DIR/deploy/infra"
              docker compose up -d --build
            else
              echo "Copied files but did not find deploy/infra under $TARGET_DIR"
            fi
            exit 0
          fi
          sleep $interval
          elapsed=$((elapsed+interval))
          echo "Waiting for upload... $elapsed/$WAIT_SECONDS"
        done
        echo "Timeout waiting for manual upload. Please scp the 'deploy' folder to $USER_DEPLOY_DIR and restart the instance or run docker compose manually."
      fi
    EOT
  }

  tags = ["agent-ai"]

  # Optional GPU accelerator block
  dynamic "guest_accelerator" {
    for_each = var.enable_gpu && var.gpu_count > 0 ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }
}

resource "google_compute_firewall" "agent_allow" {
  name    = "agent-allow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  allow {
    protocol = "tcp"
    ports    = ["8000","8001","8002"]
  }

  source_ranges = var.admin_cidrs
  target_tags    = ["agent-ai"]
}
