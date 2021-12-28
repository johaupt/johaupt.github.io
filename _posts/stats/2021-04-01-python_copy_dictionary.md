---
layout: post
blog: stats
title:  "How to Copy a Python Dictionary"
date:   2021-04-01
tags:
- python
---

*When making a copy of a dictionary in Python, most approaches make a shallow copy and the original dictionary is changed by accident. Use `copy.deepcopy()` if you need a recursive copy of all elements.*

## The problem

It's a common Python pitfall for beginners to forget that objects are mutable when copying them and to make unintended changes as a consequence. I thought I had it covered until this happend:


```python
### Don't do this ###

a = {'a_1':[1], "a_2":[2]}
b = {'b_1':[1], "b_2":[2]}
c = {'a':a,'b':b}

def f(x):
    x = x.copy() # Nice try, doesn't work
    x['a']['a_1'] = "test"
    return x

print(f(c))
print(c)
```

    {'a': {'a_1': 'test', 'a_2': [2]}, 'b': {'b_1': [1], 'b_2': [2]}}
    {'a': {'a_1': 'test', 'a_2': [2]}, 'b': {'b_1': [1], 'b_2': [2]}}


The function had an unintended side-effect even though I was purposefully using `.copy()`! 

To be clear, this is the typical example in Python beginners courses:


```python
a = [1,2,3]
b = a
b.append(1)

print(a)
print(b)
```

    [1, 2, 3, 1]
    [1, 2, 3, 1]


An easy solution it to copy element-by-element or use `.copy()`


```python
a = [1,2,3]
b = a[:] # or b = a.copy()
b.append(1)

print(a)
print(b)
```

    [1, 2, 3]
    [1, 2, 3, 1]


As I learned the hard way, the solution is not that simple for dictionaries!

## Simple assignment

As above, we'd expect just assigning a dictionary to a new variable to fail. And it does, both when we change elements of the dictionary and when we change the values of the dictionary.

We want `c` to not change because of the two changes to `d`: one to a key of `d`, the other to a value of `d`.


```python
a = {'a_1':[1], "a_2":[2]}
b = {'b_1':[1], "b_2":[2]}
c = {'a':a,'b':b}

d = c
# Assign new key
d['new'] = d.pop('b')
print(f"{d=}")
# Assign new value to an element within the dictionary
d['a']['a_1'] = 'test'
print(f"{d=}")

print(f"{c=}")
```

    d={'a': {'a_1': [1], 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    d={'a': {'a_1': 'test', 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    c={'a': {'a_1': 'test', 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}


## Element by element copy

[!] This copies the dictionary structure, but does not copy the elements themselves!

See how the new key isn't present in `c` but the change to the object in `a` is.


```python
a = {'a_1':[1], "a_2":[2]}
b = {'b_1':[1], "b_2":[2]}
c = {'a':a,'b':b}

d = {key:item for key, item in c.items()}
# Assign new key
d['new'] = d.pop('b')
print(f"{d=}")
# Assign new value to an element within the dictionary
d['a']['a_1'] = 'test'
print(f"{d=}")

print(f"{c=}")
```

    {'a': {'a_1': [1], 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    {'a': {'a_1': 'test', 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    {'a': {'a_1': 'test', 'a_2': [2]}, 'b': {'b_1': [1], 'b_2': [2]}}


## .copy()
[!] This copies the dictionary structure, but does not copy the elements themselves!


```python
a = {'a_1':[1], "a_2":[2]}
b = {'b_1':[1], "b_2":[2]}
c = {'a':a,'b':b}

d = c.copy()
# Assign new key
d['new'] = d.pop('b')
print(f"{d=}")
# Assign new value to an element within the dictionary
d['a']['a_1'] = 'test'
print(f"{d=}")

print(f"{c=}")
```

    {'a': {'a_1': 'test1', 'a_2': [2]}, 'b': {'b_1': [1], 'b_2': [2]}}
    {'a': {'a_1': 'test1', 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}


## Deepcopy

`copy.deepcopy()` copies the structure and elements recursively. `c` is unchanged after the changes to `d`


```python
import copy
```


```python
a = {'a_1':[1], "a_2":[2]}
b = {'b_1':[1], "b_2":[2]}
c = {'a':a,'b':b}

d = copy.deepcopy(c)
# Assign new key
d['new'] = d.pop('b')
print(f"{d=}")
# Assign new value to an element within the dictionary
d['a']['a_1'] = 'test'
print(f"{d=}")

print(f"{c=}")
```

    d={'a': {'a_1': [1], 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    d={'a': {'a_1': 'test', 'a_2': [2]}, 'new': {'b_1': [1], 'b_2': [2]}}
    c={'a': {'a_1': [1], 'a_2': [2]}, 'b': {'b_1': [1], 'b_2': [2]}}


## Conclusion

Be very careful when changing mutable objects within dictionaries and use `copy.deepcopy()` when necessary. Be aware that deepcopy is slow.
