---
layout: post
title: Finding Prime Numbers with Python
image: "/posts/primes_image.jpg"
tags: [Python, Primes]
---

In this post I'm going to create a function in Python that can quickly find all the Prime numbers below a given value. For example, if I input 100, it will find all the prime numbers below 100!

What is a prime number? It is a number that can only be divided wholly by itself and one so 7 is a prime number as no other numbers apart from 7 or 1 divide cleanly into it. Eight, on the other hand, is not a prime number as eight can be divided by 2 and 4 as well.

Let's go!
---

I'll begin by setting up an upper limit in the list we're going to search through.

We'll start with 20.

```py
n = 20
```

The smallest true Prime number is 2, so we'll begin our list with this number and will end it with the number that was set above as the upper bound. We use `n+1` as the set limit because the `range` upper limit is not inclusive.

Here we'll use a `set` instead of a `list`, because sets have special functions that will allow us to eliminate non-primes efficiently during our search.

```py
number_range = set(range(2, n+1))
```

If we print this variable we obtain

```py
print(number_range)
{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```

Let's also create a place where we can store any primes we discover.  A list will be perfect for this job.

```py
primes_list = []
```

We'll be using a `while loop` to iterate through our set and check for primes, but before that it is important to code up the logic and iterate manually first so that we can check if the program is working correctly step by step.


So, we have our set of numbers called **number_range**. We extract the first number from the set and want to check if it's a prime or not. If it is a prime we add it to our list called `primes_list`. If it's not, we will `pop`it out.

In fact there is a specific method in sets called `pop`! If we use pop, and assign this to the object called **prime** it will *pop* the first element from the set out of **number_range**, and into **prime**.

```py
prime = number_range.pop()
print(prime)
2
print(number_range)
{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
```

We know that the first value in our set is a prime, which is the number 2. Let's add it to our list of primes.

```py
primes_list.append(prime)
print(primes_list)
[2]
```

We will now get into the most interesting part of the algorithm. How do we check for the other numbers? We just use the number we just checked (in this case 2) and we will generate all the multiples of that number up to our upper range number (in this case 20).

Again, we will use a set, because sets allow a particular function that we will need. The range is defined as starting on 2*2 up to 20 in steps of 2. We don't need the first value as it was already been added as a prime.

```py
multiples = set(range(prime*2, n+1, prime))
```

Lets have a look at our list of multiples...

```py
print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

Now we will use the function `difference_update` which removes any values from our number range that are multiples of the number we just checked. All multiple numbers of two **are now prime numbers!**. 


Before we apply the **difference_update**, let's look at our two sets.

```py
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

**difference_update** works in a way that will update one set to only include the values that are *different* from those in a second set

To use this, we put our initial set and then apply the difference update with our multiples

```py
number_range.difference_update(multiples)
print(number_range)
>>> {3, 5, 7, 9, 11, 13, 15, 17, 19}
```

When we look at our number range, all values that were also present in the multiples set have been removed as we *know* they were not primes

Our list is reduced by half!

It is also interesing to observe that the smallest number in our range *is a prime number* as we know nothing smaller than it divides into it...and this means we can run all that logic again from the top!

Therefore, now it is the time to apply our while loop. 

The code below shows a while loop to look for the primes up to 1000.

```py
n = 1000

# number range to be checked
number_range = set(range(2, n+1))

# empty list to append discovered primes to
primes_list = []

# iterate until list is empty
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

Let's print the primes_list to have a look at what we found!

```py
print(primes_list)
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

Let's summarize our findings by using some basic stats.

```py
prime_count = len(primes_list)
largest_prime = max(primes_list)
print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
>>> There are 168 prime numbers between 1 and 1000, the largest of which is 997
```

From here we can encapsulate our `while` loop into a function.

```py
def primes_finder(n):
    
    # number range to be checked
    number_range = set(range(2, n+1))

    # empty list to append discovered primes to
    primes_list = []

    # iterate until list is empty
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
        
    prime_count = len(primes_list)
    largest_prime = max(primes_list)
    print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
```

Let's go for something large, say a million...

```py
primes_finder(1000000)
>>> There are 78498 prime numbers between 1 and 1000000, the largest of which is 999983
```

This takes just a fraction of a second!

I hoped you enjoyed learning about Primes, and one way (out of many!) to search for them using Python.

---

### Important Note: Using pop() on a Set in Python

In the real world - we would need to make a consideration around the pop() method when used on a Set as in some cases it can be a bit inconsistent.

The pop() method will usually extract the lowest element of a Set. Sets however are, by definition, unordered. The items are stored internally with some order, but this internal order is determined by the hash code of the key (which is what allows retrieval to be so fast). 

This hashing method means that we can't 100% rely on it successfully getting the lowest value. In very rare cases, the hash provides a value that is not the lowest.

Even though here, we're just coding up something fun - it is most definitely a useful thing to note when using Sets and pop() in Python in the future!

The simplest solution to force the minimum value to be used is to replace the line...

```py
prime = number_range.pop()
```

...with the lines...

```py
prime = min(sorted(number_range))
number_range.remove(prime)
```

...where we firstly force the identification of the lowest number in the number_range into our prime variable, and following that we remove it.

However, because we have to sort the list for each iteration of the loop in order to get the minimum value, it's slightly slower than what we saw with pop()!

---

Top image: "The first 25 prime numbers" | Chris @ Flickr. [https://www.flickr.com/photos/chrisinplymouth/4262775481](https://www.flickr.com/photos/chrisinplymouth/4262775481)
