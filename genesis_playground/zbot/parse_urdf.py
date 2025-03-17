import xml.etree.ElementTree as ET
import argparse

def parse_urdf_joint_limits(urdf_file_path):
    """
    Parse a URDF file and extract joint limits for joints with a <limit> tag.
    
    Args:
        urdf_file_path (str): Path to the URDF file
        
    Returns:
        dict: Mapping of joint names to their [lower, upper] limits
    """
    # Parse the XML file
    tree = ET.parse(urdf_file_path)
    root = tree.getroot()
    
    # Dictionary to store joint limits
    joint_limits = {}
    
    # Find all joint elements
    for joint in root.findall('.//joint'):
        # Get the joint name
        joint_name = joint.get('name')
        
        # Check if the joint has a limit element
        limit_elem = joint.find('./limit')
        if limit_elem is not None:
            # Extract lower and upper limits
            lower_limit = float(limit_elem.get('lower'))
            upper_limit = float(limit_elem.get('upper'))
            
            # Add to dictionary
            joint_limits[joint_name] = [lower_limit, upper_limit]
    
    return joint_limits

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse joint limits from a URDF file')
    parser.add_argument('urdf_file', type=str, help='Path to the URDF file')
    args = parser.parse_args()
    
    # Parse the URDF file
    joint_limits = parse_urdf_joint_limits(args.urdf_file)
    
    # Print the results
    print("Joint Limits:")
    for joint_name, limits in joint_limits.items():
        print(f"  {joint_name}: {limits}")