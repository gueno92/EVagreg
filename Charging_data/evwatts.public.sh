#!/bin/bash
# Check for URL expiration
now=$(date +%s)
expires=1723030845
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
if [ ! -f evwatts.public.session.csv ]; then
$program "evwatts.public.session.csv" "https://livewire-data.s3.amazonaws.com/evwatts/evwatts.public/session/evwatts.public.session.csv?response-content-disposition=%27attachment%3B%20filename%3D%22evwatts.public.session.csv%22%3B%27&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA24DU6O2GAFNN75TU%2F20240731%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240731T114045Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHQaCXVzLXdlc3QtMiJGMEQCICZfXraV3g03goW3v6KiD45XeyGBAUP94rfx%2BocFNZuYAiAfqEtQnXmbTVXNSLxajnXmZkmsznUVByf6ksz5%2BsWcbSr6AghdEAQaDDc0NzU2OTU3NTU2NCIMLbpAuFrH4RIkuC4bKtcCFdXH2Y%2Fnj1HTac7D23yqEbcNxiDZqRRUCqKGLNYorcPsixplQGWIDn2mBi4EZw3HkkJjWJQjOWs6tgENEDSo%2FEsqnuNnUVxVBDU11xLUHeqL54ywiZtSt3OzpjP8dpQzQrGNvOiFkSfdel5AMAL0zzowl2bRFSQ3jvbYYAFpVUwWbdRFRFQ1qguEVKxCStC3257sPaGyhK%2BHG54JkE%2BPHu0YMg14sovQsvNS%2F1aoTadsoyOxp2V7l0JjfLjPUuUrpdnX%2FQyhTjDop2HWXmyJFtKVh7FDQZ27ji%2B2iivpryuu%2BNA%2FdnRweXoc0km2Hw8c%2Fo8p3N2RjAAWK6W1YSbLhMbK8Fr9yudBDMRHF7KI2RFqpPjtVasarRexoS8DK0YFFpBVkBYpTiYen8Y3dzQMu8cK3NukwKMxYm99uu0mMXJ2555v58V%2FohSTeA2mu2z%2Fu8jHU8qEJTC6xai1BjqfARSXkPhJh84Nrd1kc0FwarIp%2FXpET1ziQzaT2JiBVOiRh6ZY4lwc6Fgv8LLjPtK%2F1TOmPmAfN3frIv%2FamEu8s486%2FnztChNhPWy2e0kISoz4WRDRv2aQbIvM77xqvAD1mV2bnoo2OTy4yBgXpcQtlIr0MeC9OZQjiUUuEeSyFkmt8qihELhxjrjhi04Bec5LvOGko%2FNLoMsvPEj9lkF2ug%3D%3D&X-Amz-Signature=71819141b8085418d3cffbd206d30294b0d43655c2a02a59e60fa5774fd465fb"
fi
