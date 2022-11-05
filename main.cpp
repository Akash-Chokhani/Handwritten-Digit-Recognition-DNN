#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

void init(double** w[],double* b[])
{
    for(int i=0;i<28;++i)
        b[0][i]=(i-13)/100.0;
}

int train(double** w[],double* b[],double x[],int y)
{
    int i,j,z;
    double s,p;
    double* x1=new double[28];
    double* x2=new double[10];
    double* dy=new double[10];
    
    for(j=0;j<28;++j)
    {
        s=b[0][j];
        for(i=0;i<784;++i)
            s+=w[0][i][j]*x[i];
        x1[j]=1/(1+std::exp(-1*s));
    }
    
    for(j=0;j<10;++j)
    {
        s=b[1][j];
        for(i=0;i<28;++i)
            s+=w[1][i][j]*x1[i];
        x2[j]=1/(1+std::exp(-1*s));
        dy[j]=0.04*((j==y)-x2[j])*x2[j]*(1-x2[j]);
    }
    
    for(j=0;j<28;++j)
    {
        s=0;
        for(i=0;i<10;++i)
            s+=w[1][j][i]*dy[i];
        s*=x1[j]*(1-x1[j]);
        b[0][j]+=s;
        for(i=0;i<784;++i)
            w[0][i][j]+=s*x[i];
    }
    
    for(j=0;j<10;++j)
    {
        b[1][j]+=dy[j];
        for(i=0;i<28;++i)
            w[1][i][j]+=dy[j]*x1[i];
    }
    
    p=0;
    z=-1;
    for(i=0;i<10;++i)
        if(x2[i]>p)
        {
            p=x2[i];
            z=i;
        }
    
    delete[] x1;
    delete[] x2;
    delete[] dy;
    
    return z==y;
}

bool test(double** w[],double* b[],double x[],int y)
{
    int i,j,z;
    double p,s;
    double* x1=new double[28];
    
    for(j=0;j<28;++j)
    {
        s=b[0][j];
        for(i=0;i<784;++i)
            s+=w[0][i][j]*x[i];
        x1[j]=1/(1+std::exp(-1*s));
    }
    
    p=0;
    z=-1;
    for(j=0;j<10;++j)
    {
        s=b[1][j];
        for(i=0;i<28;++i)
            s+=w[1][i][j]*x1[i];
        s=1/(1+std::exp(-1*s));
        if(s>p)
        {
            p=s;
            z=j;
        }
    }
    
    delete[] x1;
    return z==y;
}

int main()
{
    std::ifstream fi;
    int i,j,k,l,y;
    double a;
    std::string s;
    double* x=new double[784];
    
    double*** w=new double**[2];
    w[0]=new double*[784];
    w[0][0]=new double[784*28]{};
    for(i=1;i<784;++i)
        w[0][i]=&w[0][0][i*28];
    w[1]=new double*[28];
    w[1][0]=new double[28*10]{};
    for(i=1;i<28;++i)
        w[1][i]=&w[1][0][i*10];
    
    double** b=new double*[2];
    b[0]=new double[28]{};
    b[1]=new double[10]{};
    
    init(w,b);
    
    for(k=0;k<30;++k)
    {
        fi.open("mnist_train.csv");
        a=0;
        for(j=0;j<60000;++j)
        {
            std::getline(fi,s);
            y=s[0]-48;
            l=2;
            for(i=0;i<784;++i)
            {
                x[i]=0;
                for(;s[l]>47 && s[l]<58;++l)
                    x[i]=x[i]*10+s[l]-48;
                x[i]/=255;
                ++l;
            }
            a+=train(w,b,x,y);
        }
        fi.close();
        
        std::cout<<"Epoch "<<k+1<<":\nTraining Data Accuracy: "<<a/600<<'\n';
        
        fi.open("mnist_test.csv");
        a=0;
        for(j=0;j<10000;++j)
        {
            std::getline(fi,s);
            y=s[0]-48;
            l=2;
            for(i=0;i<784;++i)
            {
                x[i]=0;
                for(;s[l]>47 && s[l]<58;++l)
                    x[i]=x[i]*10+s[l]-48;
                x[i]/=255;
                ++l;
            }
            a+=test(w,b,x,y);
        }
        fi.close();
        
        std::cout<<"Test Data Accuracy: "<<a/100<<"\n\n";
    }
    
    delete[] x;
    delete[] w[0][0];
    delete[] w[1][0];
    delete[] w[0];
    delete[] w[1];
    delete[] w;
    delete[] b[0];
    delete[] b[1];
    delete[] b;
    return 0;
}