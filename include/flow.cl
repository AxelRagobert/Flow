__kernel void kernel1_outflow_computation(      
   int c,
   int r,
   __global double* Sz,
   __global double* Sh,
   __global double* So,
   double nodata,
   double dumping_factor                 
   )               
{                                         
   int i = get_global_id(0);
   int j = get_global_id(1);

   if (Sz[i * c + j] != nodata){
      int V[10];

      V[0] = i;   // central cell's row
      V[1] = j;   // central cell's column
      V[2] = i-1; // row of the neighbor at north
      V[3] = j;   // column of the neighbor at north
      V[4] = i;   // row of the neighbor at east
      V[5] = j-1; // column of the neighbor at east
      V[6] = i;   // row of the neighbor at west
      V[7] = j+1; // column of the neighbor at west
      V[8] = i+1; // row of the neighbor at south
      V[9] = j;   // column of the neighbor at south

      bool eliminated[5];
      eliminated[0] = false;
      eliminated[1] = false;
      eliminated[2] = false;
      eliminated[3] = false;
      eliminated[4] = false;

      double m; 
      double H[5];
      double average;

      int counter;
      bool again;
      
      m = Sh[i * c + j];
      
      if(m != 0){
         H[0] = Sz[i * c + j];
         for (int n=1; n<5; n++)
            H[n] = Sz[V[n * 2] * c + V[n * 2 + 1]] + Sh[V[n * 2] * c + V[n * 2 + 1]];

         do
         {
            again = false;
            counter = 0;
            average = m;
            for(int n=0; n<5; n++)
               if( eliminated[n] == false )
               {
                  average += H[n];
                  counter++;
               }

            if(counter != 0){
               average = average/counter;

               for (int n=0; n<5; n++){
                  if ( (average <= H[n]) && eliminated[n] == false)
                  {
                     eliminated[n] = true;
                     again = true;
                  }
               }
            }
         } while(again);

         double flow;
         for(int n=1; n<5; n++)
            if( eliminated[n] == false )
            {
               flow = (average - H[n]) * dumping_factor;
               So[n * r * c + i * c + j] = flow;
            }
      }
   }
}

__kernel void kernel2_mass_balance(      
   int c,
   int r,
   double nodata,
   __global double* Sz,
   __global double* Sh,
   __global double* So          
   )               
{  
                               
   int i = get_global_id(0);
   int j = get_global_id(1);               
   
   if (Sz[i * c + j] != nodata){
      int V[10];

      V[0] = i;   // central cell's row
      V[1] = j;   // central cell's column
      V[2] = i-1; // row of the neighbor at north
      V[3] = j;   // column of the neighbor at north
      V[4] = i;   // row of the neighbor at east
      V[5] = j-1; // column of the neighbor at east
      V[6] = i;   // row of the neighbor at west
      V[7] = j+1; // column of the neighbor at west
      V[8] = i+1; // row of the neighbor at south
      V[9] = j;   // column of the neighbor at south

      double h = Sh[i * c + j];
      for(int n=1; n<5; n++)
      {
         h += So[(5 - n) * r * c + V[n * 2] * c + V[n * 2 + 1]];
         h -= So[n * r * c + i * c + j];
      }
      Sh[i * c + j] = h;
   }
} 

__kernel void kernel3_outflow_reset(      
   int c,
   int r,
   double nodata,
   __global double* Sz,
   __global double* So          
   )               
{
   int i = get_global_id(0);
   int j = get_global_id(1);  

   if (Sz[i * c + j] != nodata){
      for(int n=1; n<5; n++)
         So[n * r * c + i * c + j] = 0.0;
   }
}