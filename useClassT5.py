import classT5

n = "Today, let's observe the short agenda as shown on this slide."

t = classT5.myT5("results/checkpoint-840")
print("check")
result = t.predict(n)
print(f"\033[92m{n}\033[0m") 
print("Trans")
print(f"\033[92m{result}\033[0m") 
