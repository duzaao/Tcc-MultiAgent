output "instance_name" {
  description = "Name of the created instance"
  value       = google_compute_instance.deploy_vm.name
}

output "instance_external_ip" {
  description = "External IP address of the instance"
  value       = google_compute_instance.deploy_vm.network_interface[0].access_config[0].nat_ip
}


output "instance_ip" {
  value = google_compute_instance.deploy_vm.network_interface[0].access_config[0].nat_ip
}
