#!/bin/bash
# Power Management Script for IMX415 Streamer on Rock5C
# Reduces power consumption while maintaining YOLO detection capability
#
# Usage:
#   ./power_save.sh low      # Aggressive power saving (~2-2.5W target)
#   ./power_save.sh medium   # Balanced (disable big cores, NPU at 700MHz)
#   ./power_save.sh npu_max  # LITTLE cores only, NPU at max (best YOLO perf)
#   ./power_save.sh high     # Performance mode (all cores, max NPU)
#   ./power_save.sh status   # Show current power configuration and optimizations

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# CPU paths
LITTLE_CORES="0 1 2 3"  # Cortex-A55 (efficient)
BIG_CORES="4 5 6 7"     # Cortex-A76 (power hungry)

# NPU path
NPU_FREQ="/sys/class/devfreq/fdab0000.npu"

# ISP paths
ISP_VIDEO="/dev/video0"
ISP_MEDIA="/dev/media0"
ISP_POWER_DOMAIN="/sys/devices/platform/fd8d8000.power-management/fd8d8000.power-management:power-controller/consumer:platform:fdcb0000.rkisp/power"

# Function to check ISP status
check_isp_status() {
    isp_active=0
    isp_accessible=0
    isp_in_use=0
    
    # Check if ISP device exists
    if [ -c "$ISP_VIDEO" ]; then
        isp_accessible=1
        
        # Check if any process is actively using /dev/video0 (ISP mainpath)
        # Use multiple methods for reliability
        if fuser "$ISP_VIDEO" 2>/dev/null | grep -q . || \
           lsof "$ISP_VIDEO" 2>/dev/null | grep -qv "COMMAND"; then
            isp_in_use=1
            isp_active=1
        fi
        
        # Also check if there's an active stream by trying to query format
        # (this is a lighter check - just verifies device is accessible)
        if v4l2-ctl -d "$ISP_VIDEO" --get-fmt-video >/dev/null 2>&1; then
            # Device is accessible, but check if something is actually streaming
            # If format query succeeds but no process has it open, it's just configured
            :
        fi
    fi
    
    # Check media controller for ISP pipeline existence (but don't rely on ENABLED links alone)
    # ENABLED just means configured, not actively processing
    if media-ctl -p -d "$ISP_MEDIA" 2>/dev/null | grep -qi "rkisp"; then
        isp_accessible=1
        # Only mark as active if we confirmed a process is using it
        # (already checked above with fuser/lsof)
    fi
    
    echo "$isp_active|$isp_accessible|$isp_in_use"
}

show_status() {
    echo -e "${YELLOW}=== Current Power Configuration ===${NC}"
    echo ""
    
    echo -e "${GREEN}CPU Cores Online:${NC}"
    little_online=0
    big_online=0
    
    echo -n "  LITTLE (A55): "
    for cpu in $LITTLE_CORES; do
        if [ "$(cat /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null)" == "1" ]; then
            echo -n "cpu$cpu✓ "
            little_online=$((little_online + 1))
        else
            echo -n "cpu$cpu✗ "
        fi
    done
    echo ""
    
    echo -n "  BIG (A76):    "
    for cpu in $BIG_CORES; do
        if [ "$(cat /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null)" == "1" ]; then
            echo -n "cpu$cpu✓ "
            big_online=$((big_online + 1))
        else
            echo -n "cpu$cpu✗ "
        fi
    done
    echo ""
    
    echo ""
    echo -e "${GREEN}CPU Frequencies & Governors:${NC}"
    for cpu in $LITTLE_CORES $BIG_CORES; do
        if [ "$(cat /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null)" == "1" ]; then
            freq=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq 2>/dev/null)
            gov=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null)
            min_freq=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_min_freq 2>/dev/null)
            max_freq=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_max_freq 2>/dev/null)
            echo "  cpu$cpu: $((freq/1000)) MHz ($gov) [min: $((min_freq/1000)) MHz, max: $((max_freq/1000)) MHz]"
        fi
    done
    
    echo ""
    echo -e "${GREEN}NPU (Neural Processing Unit):${NC}"
    npu_freq=$(cat $NPU_FREQ/cur_freq 2>/dev/null)
    npu_max_freq=$(cat $NPU_FREQ/max_freq 2>/dev/null)
    npu_avail_freq=$(cat $NPU_FREQ/available_frequencies 2>/dev/null)
    npu_gov=$(cat $NPU_FREQ/governor 2>/dev/null)
    echo "  Current: $((npu_freq/1000000)) MHz"
    echo "  Maximum: $((npu_max_freq/1000000)) MHz"
    echo "  Governor: $npu_gov"
    if [ -n "$npu_avail_freq" ]; then
        avail_list=$(echo $npu_avail_freq | awk '{for(i=1;i<=NF;i++) printf "%d ", $i/1000000}')
        echo "  Available: ${avail_list}MHz"
    fi
    
    echo ""
    echo -e "${GREEN}Memory Controller (DMC):${NC}"
    dmc_freq=$(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null)
    dmc_gov=$(cat /sys/class/devfreq/dmc/governor 2>/dev/null)
    echo "  Frequency: $((dmc_freq/1000000)) MHz ($dmc_gov)"
    
    echo ""
    echo -e "${GREEN}ISP (Image Signal Processor):${NC}"
    isp_status=$(check_isp_status)
    isp_active=$(echo "$isp_status" | cut -d'|' -f1)
    isp_accessible=$(echo "$isp_status" | cut -d'|' -f2)
    isp_in_use=$(echo "$isp_status" | cut -d'|' -f3)
    
    if [ "$isp_accessible" = "1" ]; then
        if [ "$isp_active" = "1" ]; then
            echo -e "  Status: ${GREEN}ACTIVE${NC} (in use)"
            echo "  Estimated power: ~0.7-1.5W"
            
            # Check what's using it
            isp_users=$(lsof "$ISP_VIDEO" "$ISP_MEDIA" 2>/dev/null | tail -n +2 | awk '{print $1}' | sort -u | tr '\n' ' ')
            if [ -n "$isp_users" ]; then
                echo "  Used by: ${isp_users}"
            fi
            
            # Check current format/resolution if available
            if v4l2-ctl -d "$ISP_VIDEO" --get-fmt-video 2>/dev/null | grep -q "Width\|Height"; then
                fmt_info=$(v4l2-ctl -d "$ISP_VIDEO" --get-fmt-video 2>/dev/null | grep -E "Width|Height|Pixel Format" | head -3)
                echo "  Current format:"
                echo "$fmt_info" | sed 's/^/    /'
            fi
        else
            echo -e "  Status: ${YELLOW}IDLE${NC} (available but not in use)"
            echo "  Power when active: ~0.7-1.5W"
        fi
        echo "  Device: $ISP_VIDEO"
        echo "  Media: $ISP_MEDIA"
    else
        echo -e "  Status: ${RED}NOT AVAILABLE${NC}"
        echo "  ISP device not accessible or not configured"
    fi
    
    echo ""
    echo -e "${YELLOW}=== Power Optimizations Status ===${NC}"
    echo ""
    
    echo -e "${GREEN}✓ Active Optimizations:${NC}"
    optimizations=0
    
    # Check BIG core optimization
    if [ $big_online -eq 0 ]; then
        echo "  ✓ BIG cores (A76) DISABLED - saves ~1-1.5W"
        optimizations=$((optimizations + 1))
    else
        echo -e "  ${RED}✗ BIG cores (A76) ENABLED - high power consumption${NC}"
    fi
    
    # Check CPU governor optimization
    all_powersave=true
    for cpu in $LITTLE_CORES; do
        if [ "$(cat /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null)" == "1" ]; then
            gov=$(cat /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null)
            if [ "$gov" != "powersave" ]; then
                all_powersave=false
                break
            fi
        fi
    done
    
    if [ "$all_powersave" = true ] && [ $little_online -gt 0 ]; then
        echo "  ✓ LITTLE cores using powersave governor - reduces idle power"
        optimizations=$((optimizations + 1))
    fi
    
    # Check NPU frequency optimization
    if [ -n "$npu_max_freq" ] && [ $npu_max_freq -lt 1000000000 ]; then
        echo "  ✓ NPU frequency capped at $((npu_max_freq/1000000))MHz - saves power while maintaining YOLO"
        optimizations=$((optimizations + 1))
    elif [ -n "$npu_max_freq" ] && [ $npu_max_freq -eq 1000000000 ] && [ $big_online -eq 0 ]; then
        echo "  ✓ NPU at max (1GHz) with BIG cores disabled - optimal YOLO performance/power tradeoff"
        optimizations=$((optimizations + 1))
    fi
    
    # Check if we're in balanced state
    if [ $big_online -eq 0 ] && [ $little_online -eq 4 ]; then
        echo "  ✓ CPU topology optimized: 4x A55 cores active (efficient)"
        optimizations=$((optimizations + 1))
    fi
    
    # Check ISP optimization
    if [ "$isp_accessible" = "1" ]; then
        if [ "$isp_active" = "0" ]; then
            echo "  ✓ ISP DISABLED (using CIF bypass) - saves ~0.7-1.5W vs ISP mode"
            optimizations=$((optimizations + 1))
        else
            echo -e "  ${YELLOW}⚠ ISP ACTIVE - adds ~0.7-1.5W but enables hardware color processing${NC}"
        fi
    fi
    
    if [ $optimizations -eq 0 ]; then
        echo -e "  ${RED}✗ No power optimizations active${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Estimated Power Consumption:${NC}"
    
    # Calculate base power
    base_power=0
    if [ $big_online -eq 0 ] && [ -n "$npu_max_freq" ] && [ $npu_max_freq -lt 600000000 ]; then
        base_power=2.25  # LOW mode average
        mode_str="LOW mode"
    elif [ $big_online -eq 0 ] && [ -n "$npu_max_freq" ] && [ $npu_max_freq -lt 800000000 ]; then
        base_power=2.75  # MEDIUM mode average
        mode_str="MEDIUM mode"
    elif [ $big_online -eq 0 ] && [ -n "$npu_max_freq" ] && [ $npu_max_freq -eq 1000000000 ]; then
        base_power=3.25  # NPU_MAX mode average
        mode_str="NPU_MAX mode"
    elif [ $big_online -gt 0 ]; then
        base_power=3.75  # HIGH mode average
        mode_str="HIGH mode"
    else
        base_power=3.0
        mode_str="Unknown mode"
    fi
    
    # Add ISP power if active
    if [ "$isp_active" = "1" ]; then
        isp_power=1.1  # Average ISP power (~0.7-1.5W range)
        total_power=$(awk "BEGIN {printf \"%.1f\", $base_power + $isp_power}")
        echo "  ~${total_power}W (${mode_str} + ISP ~+1.1W)"
    else
        echo "  ~${base_power}W (${mode_str})"
        if [ "$isp_accessible" = "1" ]; then
            echo "    (+0.7-1.5W if ISP enabled)"
        fi
    fi
}

set_low_power() {
    echo -e "${YELLOW}Setting LOW POWER mode...${NC}"
    
    # Disable all BIG cores (A76)
    echo -e "${GREEN}Disabling BIG cores (A76)...${NC}"
    for cpu in $BIG_CORES; do
        echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        echo "  cpu$cpu: disabled"
    done
    
    # Set LITTLE cores to powersave
    echo -e "${GREEN}Setting LITTLE cores to powersave...${NC}"
    for cpu in $LITTLE_CORES; do
        echo powersave > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || true
    done
    
    # Set NPU to lower frequency (500MHz - still fast for YOLO)
    echo -e "${GREEN}Setting NPU to 500MHz...${NC}"
    echo 500000000 > $NPU_FREQ/max_freq 2>/dev/null || true
    echo userspace > $NPU_FREQ/governor 2>/dev/null || true
    echo 500000000 > $NPU_FREQ/set_freq 2>/dev/null || true
    
    echo -e "${GREEN}LOW POWER mode active!${NC}"
    echo "  - Only LITTLE cores (A55) enabled"
    echo "  - NPU capped at 500MHz"
    echo "  - Expected power: ~2-2.5W"
}

set_medium_power() {
    echo -e "${YELLOW}Setting MEDIUM POWER mode...${NC}"
    
    # Disable BIG cores
    echo -e "${GREEN}Disabling BIG cores (A76)...${NC}"
    for cpu in $BIG_CORES; do
        echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        echo "  cpu$cpu: disabled"
    done
    
    # Set LITTLE cores to ondemand
    echo -e "${GREEN}Setting LITTLE cores to ondemand...${NC}"
    for cpu in $LITTLE_CORES; do
        echo ondemand > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || true
    done
    
    # Set NPU to 700MHz max
    echo -e "${GREEN}Setting NPU to 700MHz max...${NC}"
    echo rknpu_ondemand > $NPU_FREQ/governor 2>/dev/null || true
    echo 700000000 > $NPU_FREQ/max_freq 2>/dev/null || true
    
    echo -e "${GREEN}MEDIUM POWER mode active!${NC}"
    echo "  - Only LITTLE cores (A55) enabled"
    echo "  - NPU max 700MHz (dynamic)"
    echo "  - Expected power: ~2.5-3W"
}

set_npu_max_power() {
    echo -e "${YELLOW}Setting NPU MAX mode (optimal YOLO performance)...${NC}"
    
    # Disable BIG cores
    echo -e "${GREEN}Disabling BIG cores (A76)...${NC}"
    for cpu in $BIG_CORES; do
        echo 0 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        echo "  cpu$cpu: disabled"
    done
    
    # Set LITTLE cores to ondemand (balanced for web server)
    echo -e "${GREEN}Setting LITTLE cores to ondemand...${NC}"
    for cpu in $LITTLE_CORES; do
        echo ondemand > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || true
    done
    
    # Set NPU to maximum frequency (1GHz)
    echo -e "${GREEN}Setting NPU to maximum (1GHz)...${NC}"
    echo rknpu_ondemand > $NPU_FREQ/governor 2>/dev/null || true
    echo 1000000000 > $NPU_FREQ/max_freq 2>/dev/null || true
    
    echo -e "${GREEN}NPU MAX mode active!${NC}"
    echo "  - Only LITTLE cores (A55) enabled - saves CPU power"
    echo "  - NPU at maximum 1GHz - fastest YOLO inference"
    echo "  - Best balance: ~3-3.5W with optimal detection speed"
    echo "  - Perfect for YOLO-heavy workloads"
}

set_high_power() {
    echo -e "${YELLOW}Setting HIGH POWER (performance) mode...${NC}"
    
    # Enable all cores
    echo -e "${GREEN}Enabling all CPU cores...${NC}"
    for cpu in $BIG_CORES; do
        echo 1 > /sys/devices/system/cpu/cpu$cpu/online 2>/dev/null || true
        echo "  cpu$cpu: enabled"
    done
    
    # Set all cores to ondemand
    echo -e "${GREEN}Setting all cores to ondemand...${NC}"
    for cpu in $LITTLE_CORES $BIG_CORES; do
        echo ondemand > /sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_governor 2>/dev/null || true
    done
    
    # Set NPU to max
    echo -e "${GREEN}Setting NPU to max (1GHz)...${NC}"
    echo rknpu_ondemand > $NPU_FREQ/governor 2>/dev/null || true
    echo 1000000000 > $NPU_FREQ/max_freq 2>/dev/null || true
    
    echo -e "${GREEN}HIGH POWER mode active!${NC}"
    echo "  - All 8 cores enabled"
    echo "  - NPU max 1GHz"
    echo "  - Expected power: ~4W under load"
}

# Check for root
if [ "$EUID" -ne 0 ] && [ "$1" != "status" ]; then
    echo -e "${RED}Please run with sudo for power management${NC}"
    echo "Usage: sudo $0 [low|medium|npu_max|high|status]"
    exit 1
fi

case "${1:-status}" in
    low)
        set_low_power
        echo ""
        show_status
        ;;
    medium)
        set_medium_power
        echo ""
        show_status
        ;;
    npu_max)
        set_npu_max_power
        echo ""
        show_status
        ;;
    high)
        set_high_power
        echo ""
        show_status
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 [low|medium|npu_max|high|status]"
        echo ""
        echo "Modes:"
        echo "  low     - Aggressive power saving (~2-2.5W)"
        echo "            Disables BIG cores, NPU at 500MHz"
        echo "  medium  - Balanced (~2.5-3W)"
        echo "            Disables BIG cores, NPU at 700MHz"
        echo "  npu_max - Optimal YOLO performance (~3-3.5W) [RECOMMENDED]"
        echo "            Disables BIG cores, NPU at max 1GHz"
        echo "            Best detection speed while saving CPU power"
        echo "  high    - Performance (~4W)"
        echo "            All cores, NPU at 1GHz"
        echo "  status  - Show current configuration and optimizations"
        exit 1
        ;;
esac
