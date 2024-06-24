a=2;
b=2;
c=3;
d=0;
n=0;
Max=0;

M=[1 0 0;0 1 0;0 0 1];
deter=det(M)

for n=1:10
    d=a^3+b^3;
    if (d == c^3)
       Max=n
    end
end
