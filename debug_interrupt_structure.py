#!/usr/bin/env python3
"""
Debug script to understand GraphInterrupt structure
"""

try:
    from langgraph.errors import GraphInterrupt
    from langgraph.types import interrupt
    
    print("Testing GraphInterrupt structure...")
    
    # Create a test interrupt
    try:
        interrupt({"test": "data"})
    except GraphInterrupt as e:
        print(f"GraphInterrupt type: {type(e)}")
        print(f"GraphInterrupt str: {str(e)}")
        print(f"GraphInterrupt args: {e.args}")
        print(f"GraphInterrupt dir: {[attr for attr in dir(e) if not attr.startswith('_')]}")
        
        # Check if it has interrupts attribute
        if hasattr(e, 'interrupts'):
            print(f"Has interrupts: {e.interrupts}")
        else:
            print("No interrupts attribute")
            
        # Check args structure
        if e.args and len(e.args) > 0:
            first_arg = e.args[0]
            print(f"First arg type: {type(first_arg)}")
            print(f"First arg: {first_arg}")
            
            # Check if first arg has value
            if hasattr(first_arg, 'value'):
                print(f"First arg value: {first_arg.value}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()