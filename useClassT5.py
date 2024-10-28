import classT5

t = classT5.myT5("results/checkpoint-20")
print("check")
result = t.predict("Today, let's observe the short agenda as shown on this slide.")
print(result)
