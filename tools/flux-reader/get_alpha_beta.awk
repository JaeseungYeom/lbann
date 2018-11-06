#!/bin/awk -f
BEGIN {
     scalar[1]  = "BWx"
     scalar[2]  = "BT"
     scalar[3]  = "tMAXt"
     scalar[4]  = "BWn"
     scalar[5]  = "MAXpressure"
     scalar[6]  = "BAte"
     scalar[7]  = "MAXtion"
     scalar[8]  = "tMAXpressure"
     scalar[9]  = "BAt"
     scalar[10] = "Yn"
     scalar[11] = "Ye"
     scalar[12] = "Yx"
     scalar[13] = "tMAXte"
     scalar[14] = "BAtion"
     scalar[15] = "MAXte"
     scalar[16] = "tMAXtion"
     scalar[17] = "BTx"
     scalar[18] = "MAXt"
     scalar[19] = "BTn"
     scalar[20] = "BApressure"
     scalar[21] = "tMINradius"
     scalar[22] = "MINradius"
     input[1] = "shape_model_initial_modes:(4,3)"
     input[2] = "betti_prl15_trans_u"
     input[3] = "betti_prl15_trans_v"
     input[4] = "shape_model_initial_modes:(2,1)"
     input[5] = "shape_model_initial_modes:(1,0)"
}

($1 == "alpha:") {
  for (i=2; i <= NF; i++) {
    alpha[i-1] = $i;
  }
  num_alphas = NF-1;
}
($1 == "beta:") {
  for (i=2; i <= NF; i++) {
    beta[i-1] = $i;
  }
  num_betas = NF-1;
}

END {
  if (num_alphas != num_betas) {
    print "num parameters do not match"
    exit
  }
  if (num_alphas == 22) {
    for (i=1; i < num_alphas; i++) {
      printf("      { scale: %s\tbias: %s}, # %s\n", alpha[i], beta[i], scalar[i]);
    }
    printf("      { scale: %s\tbias: %s}  # %s\n", alpha[i], beta[i], scalar[i]);
  } else if (num_alphas == 5) {
    for (i=1; i < num_alphas; i++) {
      printf("      { scale: %s\tbias: %s}, # %s\n", alpha[i], beta[i], input[i]);
    }
    printf("      { scale: %s\tbias: %s}  # %s\n", alpha[i], beta[i], input[i]);
  }
}
