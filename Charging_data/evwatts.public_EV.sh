#!/bin/bash
# Check for URL expiration
now=$(date +%s)
expires=1723031871
if (( $now > $expires )); then
    echo "URLs have expired."
   exit
fi
# Check for wget/curl
if hash wget 2>/dev/null; then
    program="wget -O"
elif hash curl 2>/dev/null; then
    program="curl -o"
else
    echo "Could not find wget or curl."
    exit
fi

# Download files one by one
if [ ! -f evwatts.public.vehiclesessions.csv ]; then
$program "evwatts.public.vehiclesessions.csv" "https://livewire-ansible-prod-archive.s3.amazonaws.com/evwatts/evwatts.public/vehiclesessions/evwatts.public.vehiclesessions.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.vehiclesessions.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GE7XJRADL%2F20240731%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240731T115751Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHQaCXVzLXdlc3QtMiJGMEQCIHbUWmyapKuB4qU%2FH8afJybbdrH9xQgl85KrLRRxZ5YQAiB4j8PUztrPVIUTVV5ID3uxs0kuqPyLjpYClYePLgz1qyr6AghdEAQaDDc0NzU2OTU3NTU2NCIMxObcpHSYu3%2ByYvRuKtcC%2BQYgucpepXu87R3F32jG9okx6Yb7KxBOGChqsiJH%2F6Z7atkLanI6Y%2BJJz%2BEsxDY6gIYUEZHNy3JPXB5dJP8chUBZayjMbExDO7n3Jmh%2Fccz7KpbcI1fcboyBbExJVC7Gyy6il2ojbTF0TEJEkrND8eWyJ0%2B3ZEWTc9upTzCqQ5w4UwkEtEhi0YMd5kI2m0buIP272UjLFvxoOoWCncgS%2F7AeZ8kdZ%2B8pUvma%2FkZTnNk6zwAoKhXfDP4tF0Q9YLz87waaPGNAOTOIJul%2FQJBxY3etslSBt8cRHVIEE%2FMI0c2%2F9EkRcov11snPTu5Te8H3Lh25im694cYAkF2PWvQsySF%2BcC%2F%2BaXtd0nvJM3atAALyPp2v%2F24WSiH1WExJptDh9Rrni5Bv6Gc6Dhd%2BFnZd9JUmSLI7L5GAyKfQ9gHwKhdPtfvgqmTvqJ3%2FqBPR4J8PbPSXtfokQDC9zai1BjqfASSXDvfzHJ8iZ7j2JTzcQ6kile1foXrEeplUHH%2Fc0N6Aq%2F8aiWdrSk%2B0uew53dkWAXwMv1SNyup8iEeOBWkXzaoZObRcMjBpBp4lUS9gabIOr9BWipnFbFgENpczC4F1qn3EY5GTcMpZHgaDQaSbrA4xMHihFtP%2FhAn4rOzXgg753fn8pXpLMs9MIRyc2cKW%2FbTyvjEKaWhqoL0M7j4L1w%3D%3D&X-Amz-Signature=229fcf5c0ca339069ebbc28bb6b4da929610079b5b22b30e30f5bc31b4e0b98d"
fi
