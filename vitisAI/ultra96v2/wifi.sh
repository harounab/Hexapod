#!/bin/bash
# get the wpa conf path 
WPA_CONF=$1

if [ -z "$1" ] 
then
 echo "please provide wpa_supplicant.conf file as argument"

 else

SSID=$(sed -nE 's/^\s*SSID="(.*)"/\1/p' "$WPA_CONF")

# Extract the password
PASSWORD=$(sed -nE 's/^\s*PASSWORD="(.*)"/\1/p' "$WPA_CONF")

# Print the extracted SSID and password
echo "SSID: $SSID"
echo "Password: $PASSWORD"
# Specify the SSID and password of the Wi-Fi network

# Start the wpa_supplicant service
wpa_supplicant -B -iwlan0 -c<(wpa_passphrase "$SSID" "$PASSWORD")

# Obtain an IP address using DHCP
dhclient wlan0

# Verify the connection
ping -c 4 8.8.8.8

# Optional: Show network connection information
iwconfig wlan0
fi
